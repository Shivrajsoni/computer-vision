"""
Minimal LLaVA-style multimodal language model.

    image ──► Vision Encoder (ViT) ──► Projector (MLP) ──► image tokens ─┐
                                                                         ├─► [img | text] ──► Causal LM ──► logits
    text  ──► Tokenizer ──► Token Embedding ──► text tokens ─────────────┘

Three parts:
  1. Vision encoder : turns an image into a sequence of patch features (reuses vit.py).
  2. Projector      : maps vision features into the LLM's embedding space, so an
                      image patch looks like "just another word embedding".
  3. Language model : decoder-only transformer with causal attention. It reads the
                      image tokens as a prefix, then predicts text one token at a time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit import PatchEmbedding, ViTBlock


# 1. Vision encoder ----------------------------------------------------------
class VisionEncoder(nn.Module):
    """ViT without the classification head. Returns one feature per patch."""
    def __init__(self, img_size=224, patch_size=16, dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dim))
        self.blocks = nn.Sequential(*[ViTBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(img) + self.pos_embed   # [B, 196, vision_dim]
        return self.norm(self.blocks(x))             # [B, 196, vision_dim]


# 2. Projector ---------------------------------------------------------------
class Projector(nn.Module):
    """2-layer MLP: vision_dim -> llm_dim. This is the "bridge" between modalities."""
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                           # [B, 196, llm_dim]


# 3. Language model ----------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Same as ViT attention, but each token may only attend to earlier tokens."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # causal mask applied inside
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class DecoderBlock(nn.Module):
    """Pre-norm block: causal attention + MLP (same layout as ViTBlock)."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class LanguageModel(nn.Module):
    """Decoder-only transformer. Takes *embeddings* (not token ids) so image
    tokens and text tokens can be fed through the same stack."""
    def __init__(self, vocab_size: int, dim: int, depth: int, num_heads: int, max_len: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.Sequential(*[DecoderBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        x = embeds + self.pos_embed[:, : embeds.size(1)]
        return self.lm_head(self.norm(self.blocks(x)))   # [B, N, vocab]


# Full model -----------------------------------------------------------------
class MultimodalLLM(nn.Module):
    def __init__(self, vocab_size=32000, vision_dim=768, llm_dim=512,
                 vision_depth=12, llm_depth=8, num_heads=8, max_len=512):
        super().__init__()
        self.vision = VisionEncoder(dim=vision_dim, depth=vision_depth)
        self.projector = Projector(vision_dim, llm_dim)
        self.llm = LanguageModel(vocab_size, llm_dim, llm_depth, num_heads, max_len)

    def embed_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.projector(self.vision(img))          # [B, 196, llm_dim]

    def forward(self, img: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        img_tokens = self.embed_image(img)               # [B, P, D]   P = 196 patches
        txt_tokens = self.llm.tok_embed(input_ids)       # [B, T, D]
        seq = torch.cat([img_tokens, txt_tokens], dim=1) # [B, P+T, D] image first, then text
        logits = self.llm(seq)                           # [B, P+T, vocab]

        if labels is None:
            return logits

        # Next-token loss on text only: the image prefix has no target.
        # Logits at position i predict token i+1, so the text logits start at P-1.
        P = img_tokens.size(1)
        pred = logits[:, P - 1 : -1]                     # [B, T, vocab]
        loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), labels.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, img: torch.Tensor, input_ids: torch.Tensor, max_new_tokens: int = 20) -> torch.Tensor:
        """Greedy decoding. Image is encoded once; text grows one token per step."""
        img_tokens = self.embed_image(img)
        for _ in range(max_new_tokens):
            seq = torch.cat([img_tokens, self.llm.tok_embed(input_ids)], dim=1)
            next_id = self.llm(seq)[:, -1].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


if __name__ == "__main__":
    model = MultimodalLLM()
    img = torch.randn(2, 3, 224, 224)
    ids = torch.randint(0, 32000, (2, 16))               # e.g. tokenized "Describe this image."

    logits, loss = model(img, ids, labels=ids)
    print(f"Logits shape : {logits.shape}")              # [2, 196+16, 32000]
    print(f"Loss         : {loss.item():.3f}")           # ~ln(32000) = 10.4 at init
    print(f"Generated    : {model.generate(img, ids, 5).shape}")  # [2, 21]
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")
