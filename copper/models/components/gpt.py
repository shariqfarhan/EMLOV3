import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, reduce, repeat, einsum
from einops.layers.torch import Rearrange, Reduce


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_dim = n_dim
        self.h_dim = n_dim // n_heads

        self.keys = nn.Linear(n_dim, self.h_dim * self.n_heads)
        self.queries = nn.Linear(n_dim, self.h_dim * self.n_heads)
        self.values = nn.Linear(n_dim, self.h_dim * self.n_heads)

        self.proj = nn.Linear(n_dim, n_dim)

        self.layer_norm = nn.LayerNorm(n_dim)

        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        key = rearrange(
            self.keys(x),
            'b time (nh dim) -> nh b time dim', nh=self.n_heads
        )
        query = rearrange(
            self.queries(x),
            'b time (nh dim) -> nh b time dim', nh=self.n_heads
        )
        value = rearrange(
            self.values(x),
            'b time (nh dim) -> nh b time dim', nh=self.n_heads
        )

        # print(f"key shape: {key.shape}")
        # print(f"query shape: {query.shape}")
        # print(f"value shape: {value.shape}")

        energies = einsum(
            query,
            key,
            'nh b qt dim, nh b kt dim -> nh b qt kt'
        )

        # print(f"energies shape before mask: {energies.shape}")

        if mask is not None:
            fill_value = torch.finfo(energies.dtype).min
            # print(f"fill_value {fill_value}")
            # print(f"mask shape {mask.shape}")
            energies = energies.masked_fill(mask, fill_value)

        # print(f"energies shape after mask: {energies.shape}")

        attn = F.softmax(energies, dim=-1)

        # print(f"attn shape: {attn.shape}")

        attn = self.attn_dropout(attn)

        out = einsum(
            attn,
            value,
            'nh b qt kt, nh b kt dim -> nh b qt dim'
        )

        # print(f"out shape before rearrange: {out.shape}")

        out = rearrange(
            out,
            'nh b vt dim -> b vt (nh dim)'
        )

        # print(f"out shape after rearrange: {out.shape}")

        out = self.proj(out)

        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()

        self.fn = fn

    def forward(self, x, **kwargs):
        res = x

        out = self.fn(x, **kwargs)

        out += res

        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size = 768, expansion = 4, drop_p = 0.):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

class GPTDecoderBlock(nn.Module):
    def __init__(
        self,
        emb_size = 768,
        drop_p = 0.,
        forward_expansion = 4,
        forward_drop_p = 0,
        n_heads=4
    ):
        super(GPTDecoderBlock, self).__init__()

        self.ln = nn.LayerNorm(emb_size)
        self.mha = MultiHeadAttention(n_heads=n_heads, n_dim=emb_size, dropout=drop_p)
        self.drop = nn.Dropout(drop_p)

        self.out_block = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
        )

    def forward(self, x, mask = None):
        residual = x

        out = self.ln(x)
        out = self.mha(out, mask)
        out = self.drop(out)
        out = x + out
        out = self.out_block(out)

        return out

class GPT(nn.Module):
    def __init__(self, block_size, n_embed, n_heads=4, n_blocks=4, drop_p=0):
        super(GPT, self).__init__()

        self.block_size = block_size
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.n_blocks = n_blocks
        vocab_size_gpt = 100277

        self.token_embedding_table = nn.Embedding(vocab_size_gpt, self.n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)
        # Use n_blocks to create the desired number of decoder blocks
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(emb_size=self.n_embed, n_heads=n_heads, drop_p=drop_p) for _ in range(self.n_blocks)
        ])
        self.ln = nn.LayerNorm(self.n_embed)
        self.ffwd = FeedForwardBlock(self.n_embed)
        self.lm_head = nn.Linear(self.n_embed, vocab_size_gpt)

        # query: what am i looking for?
        # key: what do i contain?

    def get_parameters(self):
        return {
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
            "n_embed": self.n_embed,
            "n_heads": self.n_heads,
            "drop_p": self.drop_p
        }

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

if __name__ == "__main__":
    _ = GPT()