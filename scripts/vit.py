import math
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Subset
from torchvision import datasets, transforms

class PatchEmbed(nn.Module):
    def __init__(self , img_size = 32 , patch = 4 , dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3,dim , kernel_size=patch , stride=patch)
        self.num_patches = (img_size // patch) ** 2 

    def forward(self ,x):
        return self.proj(x).flatten(2).transpose(1,2)

class Attention(nn.Module):
    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)   # (B, h, N, N)
        attn = self.drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class ViT(nn.Module):
    def __init__(self,img_size = 32,patch=4,num_classes=10,dim=192,depth=6,heads=3,drop=0.1):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, dim)
        n = self.patch.num_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.drop = nn.Dropout(drop)
        self.blocks = nn.Sequential(*[Block(dim, heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.blocks(self.drop(x))
        return self.head(self.norm(x[:, 0]))      # classify from CLS token


if __name__ == "__main__":
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # uses existing data/cifar-10-batches-py, no download
    # random subset of CIFAR-10 so a full run finishes in a few minutes
    TRAIN_N, TEST_N = 10000, 2000
    g = torch.Generator().manual_seed(0)
    train_ds = datasets.CIFAR10("data", train=True, transform=train_tf)
    test_ds = datasets.CIFAR10("data", train=False, transform=test_tf)
    train_ds = Subset(train_ds, torch.randperm(len(train_ds), generator=g)[:TRAIN_N].tolist())
    test_ds = Subset(test_ds, torch.randperm(len(test_ds), generator=g)[:TEST_N].tolist())
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = ViT(dim=128, depth=4, heads=4).to(device)   # smaller model for the subset
    EPOCHS = 10
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    start = time.time()
    losses, accs = [], []
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval(); correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        losses.append(loss.item()); accs.append(correct / len(test_dl.dataset))
        print(f"epoch {ep+1}  loss {loss.item():.3f}  test acc {correct / len(test_dl.dataset):.4f}  {time.time() - start:.0f}s")

    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(range(1, EPOCHS + 1), losses); ax[0].set(title="Train loss (last batch)", xlabel="epoch")
    ax[1].plot(range(1, EPOCHS + 1), accs); ax[1].axhline(0.1, ls="--", c="gray", label="chance")
    ax[1].set(title="Test accuracy", xlabel="epoch", ylim=(0, 1)); ax[1].legend()
    fig.tight_layout(); fig.savefig("outputs/vit_training.png", dpi=130)
