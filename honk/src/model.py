"""Small CNN keyword spotter (honk cnn-trad-pool2) with a swappable sequence-aggregation head.

    MFCC (101 x 40) -> conv(20x8) -> max-pool(2x2) -> conv(10x4)  => 32 x 13 feature map
    -> 416 frame vectors (64-d) -> Linear -> 128-d frames -> AGGREGATION -> 128-d -> Linear -> 32 logits

aggregation:
    baseline      no frame sequence: flatten the whole feature map, Linear(26624 -> 128)
    att_one_step  Bahdanau-style one-step attention pooling over the frames
    att_pure      multi-head dot-product attention against 8 learned patterns (nn.MultiheadAttention),
                  residual add, then the same attention pooling
    mhn           Modern Hopfield network: frames are queries, 8 learned patterns are the memory
                  (K = V), ``steps`` update iterations, residual add, then attention pooling
    mhnpl         HopfieldPooling: 8 learned queries attend to the frames (K = V = frames),
                  ``steps`` update iterations, then attention pooling over the 8 outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hflayers import Hopfield, HopfieldPooling

AGGREGATIONS = ("baseline", "att_one_step", "att_pure", "mhn", "mhnpl")
N_LABELS = 32
EMBED_DIM = 128
N_PATTERNS = 8
N_HEADS = 4
# Hopfield inverse temperature (beta) per aggregation, for steps >= 1
HOPFIELD_BETA = {"mhn": 1.0, "mhnpl": 0.25}
# Hopfield layers without hflayers' default internal normalisation/projection extras, so MHN and
# MHNPL differ from dot-product attention only by beta and the iterative update.
HOPFIELD_PLAIN = dict(
    normalize_stored_pattern=False, normalize_stored_pattern_affine=False,
    normalize_state_pattern=False, normalize_state_pattern_affine=False,
    normalize_pattern_projection=False, normalize_pattern_projection_affine=False, batch_first=True)


def truncated_normal_(tensor, std=0.01):
    """TensorFlow-style initialiser: N(0, std) clipped at 2 std (resampled)."""
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class SpeechModel(nn.Module):
    def __init__(self, aggregation, steps=1, supcon_dim=0, in_height=101, in_width=40):
        super().__init__()
        assert aggregation in AGGREGATIONS, aggregation
        self.aggregation = aggregation
        self.conv1 = nn.Conv2d(1, 64, (20, 8))
        self.conv2 = nn.Conv2d(64, 64, (10, 4))
        for conv in (self.conv1, self.conv2):
            truncated_normal_(conv.weight.data)
            conv.bias.data.zero_()
        self.pool1 = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            fmap = self._conv(torch.zeros(1, in_height, in_width))
        n_frames, channels = fmap.shape[2] * fmap.shape[3], fmap.shape[1]

        if aggregation == "baseline":
            self.embedding_proj = nn.Linear(n_frames * channels, EMBED_DIM)
        else:
            self.embedding_proj = nn.Linear(channels, EMBED_DIM)
            # attention pooling shared by all aggregations: softmax(w^T tanh(W h)) weighted sum
            self.frame_pool_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.frame_pool_score = nn.Linear(EMBED_DIM, 1, bias=False)
            self.frame_pool_norm = nn.LayerNorm(EMBED_DIM)
        if aggregation == "att_pure":
            self.patterns = nn.Parameter(torch.empty(1, N_PATTERNS, EMBED_DIM).normal_(0.0, 0.02))
            self.mha = nn.MultiheadAttention(EMBED_DIM, N_HEADS, batch_first=True)
        elif aggregation == "mhn":
            self.patterns = nn.Parameter(torch.empty(1, N_PATTERNS, EMBED_DIM).normal_(0.0, 0.02))
            self.hopfield = Hopfield(
                input_size=EMBED_DIM, hidden_size=EMBED_DIM // N_HEADS, output_size=EMBED_DIM,
                num_heads=N_HEADS, scaling=HOPFIELD_BETA["mhn"] if steps > 0 else 0.5,
                update_steps_max=steps, **HOPFIELD_PLAIN)
        elif aggregation == "mhnpl":
            self.hopfield = HopfieldPooling(
                input_size=EMBED_DIM, hidden_size=EMBED_DIM // N_HEADS, output_size=EMBED_DIM,
                num_heads=N_HEADS, scaling=HOPFIELD_BETA["mhnpl"], quantity=N_PATTERNS,
                update_steps_max=steps, trainable=True, **HOPFIELD_PLAIN)

        self.output = nn.Linear(EMBED_DIM, N_LABELS)
        # projection head for the SupCon loss (only when used)
        self.supcon_head = (nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(),
                                          nn.Linear(EMBED_DIM, supcon_dim)) if supcon_dim else None)

    def _conv(self, x):
        x = self.dropout(F.relu(self.conv1(x.unsqueeze(1))))
        x = self.pool1(x)
        return self.dropout(F.relu(self.conv2(x)))

    def _pool(self, frames):
        weights = torch.softmax(
            self.frame_pool_score(torch.tanh(self.frame_pool_proj(frames))).squeeze(-1), dim=1)
        pooled = (frames * weights.unsqueeze(-1)).sum(dim=1)
        return self.frame_pool_norm(pooled)

    def embed(self, x):
        """(B, 101, 40) MFCC -> (B, 128) utterance embedding."""
        fmap = self._conv(x)
        if self.aggregation == "baseline":
            return self.embedding_proj(fmap.flatten(1))
        frames = fmap.permute(0, 2, 3, 1).reshape(fmap.size(0), -1, fmap.size(1))
        frames = self.dropout(self.embedding_proj(frames))
        if self.aggregation == "att_one_step":
            return self._pool(frames)
        if self.aggregation == "att_pure":
            context, _ = self.mha(frames, self.patterns.expand(len(frames), -1, -1),
                                  self.patterns.expand(len(frames), -1, -1), need_weights=False)
            return self._pool(frames + context)
        if self.aggregation == "mhn":
            patterns = self.patterns.expand(len(frames), -1, -1)
            return self._pool(frames + self.hopfield((patterns, frames, patterns)))
        pooled = self.hopfield(frames).view(len(frames), N_PATTERNS, EMBED_DIM)  # mhnpl
        return self._pool(pooled)

    def forward(self, x, x_noisy=None, return_embedding=False):
        embedding = self.embed(x)
        logits = self.output(self.dropout(embedding))
        if return_embedding or x_noisy is not None:
            noisy = self.embed(x_noisy) if x_noisy is not None else None
            return logits, embedding, noisy
        return logits
