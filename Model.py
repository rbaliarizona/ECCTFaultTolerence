"""
@author: Yoni Choukroun, choukroun.yoni@gmail.com
Error Correction Code Transformer
https://arxiv.org/abs/2203.14966
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from Codes import sign_to_bin, bin_to_sign
import time


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers) // 2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout, use_norm):
        super(SublayerConnection, self).__init__()
        if use_norm:
            self.norm = LayerNorm(size)
        else:
            self.norm = nn.Identity()  # Pass-through layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, sublayer_norm=True):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, sublayer_norm), 2)
        self.size = size
        self.droupout = dropout

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

    def update_sublayer_norm(self, sublayer_norm):
        """Update the sublayer normalization setting."""
        self.sublayer = clones(SublayerConnection(self.size, self.droupout, sublayer_norm), 2)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


############################################################


class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        ####
        code = args.code
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_model * 4, dropout)

        self.src_embed = torch.nn.Parameter(
            torch.empty((code.n + code.pc_matrix.size(0), args.d_model))
        )
        self.decoder = Encoder(
            EncoderLayer(args.d_model, c(attn), c(ff), dropout, True), args.N_dec,
        )
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)]
        )
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)

        self.get_mask(code)
        logging.info(f"Mask:\n {self.src_mask}")
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome):
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        emb = self.decoder(emb, self.src_mask)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy_with_logits(
            z_pred, sign_to_bin(torch.sign(z2))
        )
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred

    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            mask_size = code.n + code.pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~(mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        src_mask = build_mask(code)
        mask_size = code.n + code.pc_matrix.size(0)
        a = mask_size**2
        logging.info(
            f"Self-Attention Sparsity Ratio={100 * torch.sum((src_mask).int()) / a:0.2f}%, Self-Attention Complexity Ratio={100 * torch.sum((~src_mask).int())//2 / a:0.2f}%"
        )
        self.register_buffer("src_mask", src_mask)

class ECC_TransformerNoNorm(ECC_Transformer):

    def __init__(self, args, dropout=0):
        super().__init__(args, dropout)
        for layer in self.decoder.layers:
            layer.update_sublayer_norm(False)


class GRUSyndromeReduce(nn.Module):
    """
    x: (B, T, m), mask: (B, T) -> (B, m)
    Uses GRU hidden size = 2*m, then projects back to m.
    """
    def __init__(self, m, num_layers=1):
        super().__init__()
        self.m = m
        self.h =  m
        self.gru  = nn.GRU(input_size=m, hidden_size=self.h,
                           num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(self.h + m, m)  # <-- widened for skip concat

        # (optional, good defaults)
        for name, p in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, mask):
        # x: (B,T,m), mask True for valid steps
        y, _ = self.gru(x)                                 # (B,T,2m)
        last = mask.long().sum(1).clamp_min(1) - 1         # (B,)
        bidx = torch.arange(x.size(0), device=x.device)
        h_T = y[bidx, last]                                    # (B,m)
        s_T = x[bidx, last]                                    # (B,m)
        fused = torch.cat([h_T, s_T], dim=-1)                  # (B, 2m)
        return self.proj(fused)                                # (B,m)

    def loss(self, synd_pred, final_synd):
        loss = F.binary_cross_entropy_with_logits(
            synd_pred, sign_to_bin(torch.sign(final_synd))
        )
        return loss
    

class ECCTransformerWithSyndromeHistory(ECC_Transformer):
    """
    Extends ECC_Transformer to handle a *history* of syndromes.
    Forward expects:
        magnitude:      (B, n)
        syndrome_hist:  (B, T_max, m)   # m = code.pc_matrix.size(0)
        mask:           (B, T_max) bool # True for valid timesteps
    It reduces the history to one m-dim vector using a tiny Transformer,
    then proceeds exactly like the base decoder with [magnitude || reduced_syndrome].
    """

    def __init__(self, args, heads_hist=1, layers_hist=2):
        super().__init__(args, dropout=0)  # keep base init as-is
        code = args.code
        self.n = code.n
        self.m = code.pc_matrix.size(0)
        # Order-aware reducer for syndrome history (d_in == d_out == m)
        self.syndrome_reduce = GRUSyndromeReduce(
            m=self.m, num_layers=1
        )

    def forward(self, magnitude, syndrome_hist, mask):
        """
        magnitude:     (B, n)
        syndrome_hist: (B, T_max, m)
        mask:          (B, T_max) True=valid
        returns: logits for z (B, n)
        """
        # 1) Reduce the variable-length history to one m-dim vector
        s_reduced = self.syndrome_reduce(syndrome_hist, mask)  # (B, m)

        # 2) Concatenate with magnitude to form the d = n+m token
        x = torch.cat([magnitude, s_reduced], dim=-1)  # (B, n+m)

        # 3) Use the original ECC_Transformer pipeline
        emb = x.unsqueeze(-1)  # (B, d, 1)
        emb = self.src_embed.unsqueeze(0) * emb  # (B, d, d_model)
        emb = self.decoder(emb, self.src_mask)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))


class ECCTransformerWithSimpleFrontMLP(ECC_Transformer):
    """ECC_Transformer + 1-hidden-layer GELU MLP front (no residual)."""

    def __init__(self, args, hidden_dim=None):
        super().__init__(
            args
        )  # builds backbone: src_embed, decoder, out_fc, mask
        d = args.code.n + args.code.pc_matrix.size(0)
        h = hidden_dim or d
        self.front = nn.Sequential(
            nn.Linear(d, h, bias=True),
            nn.GELU(),
            nn.Linear(h, d, bias=True),
        )

    def forward(self, magnitude, syndrome):
        x = torch.cat([magnitude, syndrome], dim=-1)  # [B, d]
        x = self.front(x)  # replace input (no residual)
        emb = x.unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        emb = self.decoder(emb, self.src_mask)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))


if __name__ == "__main__":
    pass
