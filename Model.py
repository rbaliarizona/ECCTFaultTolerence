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
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


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
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
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
        ff = PositionwiseFeedForward(args.d_model, args.d_model*4, dropout)

        self.src_embed = torch.nn.Parameter(torch.empty(
            (code.n + code.pc_matrix.size(0), args.d_model)))
        self.decoder = Encoder(EncoderLayer(
            args.d_model, c(attn), c(ff), dropout), args.N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)

        self.get_mask(code)
        logging.info(f'Mask:\n {self.src_mask}')
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
            z_pred, sign_to_bin(torch.sign(z2)))
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
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask
        src_mask = build_mask(code)
        mask_size = code.n + code.pc_matrix.size(0)
        a = mask_size ** 2
        logging.info(
            f'Self-Attention Sparsity Ratio={100 * torch.sum((src_mask).int()) / a:0.2f}%, Self-Attention Complexity Ratio={100 * torch.sum((~src_mask).int())//2 / a:0.2f}%')
        self.register_buffer('src_mask', src_mask)



class NoisySyndromeECCTDecoder(nn.Module):
    def __init__(self, ecct_model, P, syndrome_sigma=0.5, readout_sigma=0.1, max_steps=5, stop_hidden=64, stop_threshold=0.5):
        super().__init__()
        self.ecct = ecct_model
        self.P = P.float()  # [m, n]
        self.syndrome_sigma = syndrome_sigma  # AWGN on syndrome
        self.readout_sigma = readout_sigma    # AWGN on y after syndrome
        self.max_steps = max_steps
        self.stop_threshold = stop_threshold

        n = P.shape[1]
        m = P.shape[0]

        self.controller = nn.Sequential(
            nn.Linear(n + m, stop_hidden),
            nn.ReLU(),
            nn.Linear(stop_hidden, 1)
        )

    def noisy_syndrome(self, y):
        y_b = (y < 0).float()
        s = (self.P @ y_b.T).T % 2
        s = 1 - 2 * s
        return s + torch.randn_like(s) * self.syndrome_sigma

    def degrade_y(self, y):
        return y + torch.randn_like(y) * self.readout_sigma

    def forward(self, y_init, x_clean):
        """
        Iteratively decode using ECCT and learned stopping criterion.
        Returns final y and z_pred from the step where stopping occurred.
        """
        y = y_init.clone()
        batch_size = y.shape[0]
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=y.device)

        final_z = torch.zeros_like(y)
        final_y = y.clone()

        for t in range(self.max_steps):
            s = self.noisy_syndrome(y)
            y = self.degrade_y(y)

            mag = torch.abs(y)
            inp = torch.cat([mag, s], dim=1)

            z_pred = self.ecct(mag, s)
            stop_logit = self.controller(inp)
            stop_prob = torch.sigmoid(stop_logit).squeeze(1)

            should_stop = (stop_prob > self.stop_threshold) & (~stopped)

            final_z[should_stop] = z_pred[should_stop]
            final_y[should_stop] = y[should_stop]
            stopped = stopped | should_stop

            if stopped.all():
                break

        # Fallback: if nothing stopped
        if not stopped.all():
            final_z[~stopped] = z_pred[~stopped]
            final_y[~stopped] = y[~stopped]

        return final_y, final_z

    def loss(self, z_pred, y_final, x_clean):
        """
        Supervise z_pred so that sign(z_pred * y_final) ≈ x_clean
        """
        z_mul = (y_final * bin_to_sign(x_clean))
        loss = F.binary_cross_entropy_with_logits(z_pred, sign_to_bin(torch.sign(z_mul)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y_final)))
        return loss, x_pred


############################################################
############################################################




import torch
import torch.nn as nn
import torch.nn.functional as F
# relies on your existing imports from Codes:
# from Codes import sign_to_bin, bin_to_sign

class NoisySyndromeRNNECCTDecoder(nn.Module):
    """
    RNN wrapper around ECCT with learned halting via a straight-through gate.
    - At each step t:
        s_t = noisy_syndrome(y_t)              # AWGN on syndrome (syndrome_sigma)
        y_{t+1} = y_t + N(0, readout_sigma^2)  # readout noise after measurement
        z_t = ECCT(|y_{t+1}|, s_t)             # logits for per-bit flip
        h_{t+1} = GRU([|y_{t+1}|, s_t], h_t)   # controller state
        p_stop_t = sigmoid(W h_{t+1})
        gate_t = ST_Bernoulli(p_stop_t, threshold)  # hard forward, soft gradient
        if first gate_t=1 for a sample: take (y_{t+1}, z_t) as final outputs
    - forward() returns (y_final, z_pred_final). No loss is computed here.
    - loss() identical to your current formulation (moving target).
    """
    def __init__(self,
                 ecct_model,
                 P,
                 syndrome_sigma=0.5,
                 readout_sigma=0.1,
                 max_steps=5,
                 stop_hidden=64,
                 stop_threshold=0.5):
        super().__init__()
        self.ecct = ecct_model
        self.P = P.float()              # [m, n]
        self.syndrome_sigma = syndrome_sigma
        self.readout_sigma = readout_sigma
        self.max_steps = max_steps
        self.stop_threshold = stop_threshold

        n = P.shape[1]
        m = P.shape[0]
        self.rnn_cell = nn.GRUCell(input_size=n + m, hidden_size=stop_hidden)
        self.stop_head = nn.Linear(stop_hidden, 1)

    # ----- channel / measurement models -----
    def noisy_syndrome(self, y):
        # Binary map from current y, then H*yb (mod 2) → {±1} → add AWGN
        y_b = (y < 0).float()
        s = (self.P @ y_b.T).T % 2.0
        s = 1.0 - 2.0 * s
        return s + torch.randn_like(s) * self.syndrome_sigma

    def degrade_y(self, y):
        return y + torch.randn_like(y) * self.readout_sigma

    # ----- straight-through halting gate -----
    @staticmethod
    def _st_hard_sigmoid_gate(prob, threshold, force_stop_mask=None):
        """
        prob: [B] in (0,1), threshold: scalar
        force_stop_mask: optional bool [B], forces gate=1 for those entries (hard),
                         but keeps soft gradient via straight-through trick.
        Returns: gate (hard forward, soft gradient), hard_mask (bool) used this step.
        """
        hard = (prob > threshold).float()
        if force_stop_mask is not None:
            hard = torch.where(force_stop_mask, torch.ones_like(hard), hard)
        # straight-through: hard + soft - soft.detach()
        gate = hard + prob - prob.detach()
        return gate, hard.bool()

    # ----- main unroll -----
    def forward(self, y_init, x_clean):
        """
        Returns:
            y_final: [B, n] degraded y at the chosen step
            z_pred: [B, n] ECCT logits at the chosen step
        """
        device = y_init.device
        B, n = y_init.shape
        y = y_init.clone()

        # controller state
        h = torch.zeros(B, self.rnn_cell.hidden_size, device=device)

        # track which samples have already stopped
        stopped = torch.zeros(B, dtype=torch.bool, device=device)

        # running “chosen” outputs (initialized to zeros, then filled once)
        y_final = torch.zeros_like(y)
        z_final = torch.zeros_like(y)
        for t in range(self.max_steps):
            # 1) measure syndrome on current y, then degrade y
            s = self.noisy_syndrome(y)
            y = self.degrade_y(y)  # model readout corruption right after measurement

            # 2) ECCT decode on (|y|, s) to get z logits
            mag = torch.abs(y)
            z_t = self.ecct(mag, s)  # logits, shape [B, n]; ECCT expects (magnitude, syndrome). :contentReference[oaicite:3]{index=3}

            # 3) controller update & halting probability
            ctrl_in = torch.cat([mag, s], dim=1)
            h = self.rnn_cell(ctrl_in, h)                 # GRUCell
            p_stop = torch.sigmoid(self.stop_head(h)).squeeze(1)  # [B]

            # 4) build ST halting gate; force stop on last step for any still-running sample
            force_last = (~stopped) & (t == self.max_steps - 1)
            gate, hard_now = self._st_hard_sigmoid_gate(p_stop, self.stop_threshold, force_last)

            # only first stop per sample should be taken
            take_mask = ((~stopped).float() * gate).unsqueeze(1)  # [B,1], hard in fwd, soft in grad
            y_final = take_mask * y + (1.0 - take_mask) * y_final
            z_final = take_mask * z_t + (1.0 - take_mask) * z_final
            stopped = stopped | hard_now

            if stopped.all():
                break

        return y_final, z_final

    # ----- same loss you use now (moving target supervision for z) -----
    def loss(self, z_pred, y_final, x_clean):
        """
        BCE-with-logits against target z* = sign(y_final) * sign(x_clean)
        (implemented via bin targets as in your current code).
        Also returns x_pred for BER/FER.
        """
        z_mul = (y_final * bin_to_sign(x_clean))
        loss = F.binary_cross_entropy_with_logits(z_pred, sign_to_bin(torch.sign(z_mul)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y_final)))
        return loss, x_pred



if __name__ == '__main__':
    pass
