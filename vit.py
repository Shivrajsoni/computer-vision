import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Slices [B, 3, 224, 224] into 196 patches of 16x16 and projects to dim=768."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # (224 // 16)^2 = 196
        
        # Single strided Conv2d performs both patch slicing and linear projection!
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224] -> proj -> [B, 768, 14, 14]
        x = self.proj(x)
        # Flatten spatial dimensions (14*14 = 196) and transpose to [B, 196, 768]
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads          # 768 // 12 = 64
        self.scale = self.head_dim ** -0.5        # 1 / sqrt(64) = 0.125

        # Single fused Linear layer computes Q, K, and V simultaneously
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # [B, 197, 768]

        # 1. Compute Q, K, V and split into 12 heads: [3, B, 12, 197, 64]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Scaled Dot-Product Attention: (Q @ K^T) / sqrt(d_k) -> Shape: [B, 12, 197, 197]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 3. Multiply by Values V and recombine heads -> Shape: [B, 197, 768]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ViTBlock(nn.Module):
    """Standard Pre-Norm Transformer Encoder Block: MHSA + 4x MLP."""
    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),  # 768 -> 3072
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)   # 3072 -> 768
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm Residual Connection 1: Attention (Spatial Token Mixing)
        x = x + self.attn(self.norm1(x))
        # Pre-Norm Residual Connection 2: MLP (Per-Token Channel Mixing)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token [1, 1, 768] and 1D Positional Embedding [1, 197, 768]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Stack 12 Transformer Encoder Blocks
        self.blocks = nn.Sequential(*[
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                        # [B, 196, 768]

        # Prepend [CLS] token across the batch -> [B, 197, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Inject 1D Positional Embeddings via element-wise addition
        x = x + self.pos_embed                         # [B, 197, 768]

        # Pass through 12 Transformer blocks
        x = self.blocks(x)                             # [B, 197, 768]
        x = self.norm(x)

        # Extract ONLY the 0-th token ([CLS]) for classification!
        cls_output = x[:, 0]                           # [B, 768]
        return self.head(cls_output)                   # [B, 1000]


if __name__ == "__main__":
    vit = VisionTransformer()
    dummy_img = torch.randn(2, 3, 224, 224)
    logits = vit(dummy_img)
    print(f"ViT-B/16 Output Shape : {logits.shape}")  # [2, 1000]
    print(f"Total Parameters      : {sum(p.numel() for p in vit.parameters()):,}")  # ~86.5M params