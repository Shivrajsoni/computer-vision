"""
Minimal CLIP from scratch, overfit on one batch, then zero-shot classify.

    image ──► CNN encoder ──► linear ──► L2-norm ─┐
                                                  ├─► cosine sims * exp(t) ──► symmetric CE
    text  ──► Transformer ──► linear ──► L2-norm ─┘

Zero-shot: embed one caption per class ("a photo of a cat"), pick the class whose
caption is closest to the image embedding. No classifier head, no class labels in the model.
"""
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)


# 1. Image encoder: small CNN -> pooled feature -> projection into the shared space
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.proj = nn.Linear(256, embed_dim, bias=False)

    def forward(self, x):
        return self.proj(self.features(x).mean(dim=(2, 3)))


# 2. Text encoder: token + pos embedding -> transformer -> feature at the EOS token -> projection
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, ctx_len, width=128, heads=4, layers=2, embed_dim=128):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, width)
        self.pos = nn.Parameter(torch.randn(ctx_len, width) * 0.01)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim, bias=False)
        mask = torch.triu(torch.full((ctx_len, ctx_len), float("-inf")), diagonal=1)  # causal, like CLIP
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, ids):
        x = self.tok(ids) + self.pos[: ids.shape[1]]
        x = self.ln(self.tf(x, mask=self.mask[: ids.shape[1], : ids.shape[1]]))
        eos = ids.argmax(dim=-1)                      # EOS has the highest token id
        return self.proj(x[torch.arange(len(ids)), eos])


# 3. CLIP: two encoders + learnable temperature
class CLIP(nn.Module):
    def __init__(self, vocab_size, ctx_len, embed_dim=128):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, ctx_len, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, x):
        return F.normalize(self.image_encoder(x), dim=-1)

    def encode_text(self, ids):
        return F.normalize(self.text_encoder(ids), dim=-1)

    def forward(self, images, ids):
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * self.encode_image(images) @ self.encode_text(ids).T   # [B_img, B_txt]


def clip_loss(logits, labels):
    """Symmetric InfoNCE. Same-class pairs in a batch are all positives, so the target
    is a row-normalized same-class mask instead of the identity matrix."""
    pos = (labels[:, None] == labels[None, :]).float()
    target = pos / pos.sum(dim=1, keepdim=True)
    loss_i = -(target * logits.log_softmax(dim=1)).sum(1).mean()
    loss_t = -(target * logits.T.log_softmax(dim=1)).sum(1).mean()
    return (loss_i + loss_t) / 2


# 4. Tiny word-level tokenizer over the prompt vocabulary
class Tokenizer:
    def __init__(self, class_names):
        words = sorted({w for c in class_names for w in self.prompt(c).split()})
        self.stoi = {"<pad>": 0, **{w: i + 1 for i, w in enumerate(words)}}
        self.stoi["<eos>"] = len(self.stoi)
        self.ctx_len = max(len(self.prompt(c).split()) for c in class_names) + 1
        self.vocab_size = len(self.stoi)

    @staticmethod
    def prompt(c):
        return f"a photo of a {c}"

    def __call__(self, texts):
        out = torch.zeros(len(texts), self.ctx_len, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [self.stoi[w] for w in t.split()] + [self.stoi["<eos>"]]
            out[i, : len(ids)] = torch.tensor(ids)
        return out


@torch.no_grad()
def zero_shot_acc(model, tok, images, labels, class_names):
    model.eval()
    text_emb = model.encode_text(tok([tok.prompt(c) for c in class_names]).to(device))   # [C, D]
    pred = (model.encode_image(images.to(device)) @ text_emb.T).argmax(dim=1).cpu()
    return (pred == labels).float().mean().item(), pred


def plot_results(hist, sim_before, sim_after, labels, cm, class_names, test_acc, out="outputs"):
    os.makedirs(out, exist_ok=True)

    # 1. Training curves: loss and zero-shot accuracy on the train batch
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["step"], hist["loss"]); ax[0].set(title="Contrastive loss", xlabel="step", ylabel="loss")
    ax[1].plot(hist["step"], hist["acc"]); ax[1].axhline(0.1, ls="--", c="gray", label="chance")
    ax[1].set(title="Zero-shot acc (train batch)", xlabel="step", ylim=(0, 1.02)); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{out}/clip_training.png", dpi=130); plt.close(fig)

    # 2. Image-text similarity matrix, images sorted by class, before vs after training
    order = labels.argsort()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, sim, name in zip(ax, (sim_before, sim_after), ("Before training", "After training")):
        im = a.imshow(sim[order].cpu(), aspect="auto", cmap="viridis")
        a.set(title=name, xlabel="class caption", ylabel="image (sorted by class)")
        a.set_xticks(range(len(class_names))); a.set_xticklabels(class_names, rotation=90, fontsize=7)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{out}/clip_similarity.png", dpi=130); plt.close(fig)

    # 3. Confusion matrix on unseen test images (rows = true, cols = predicted)
    fig, a = plt.subplots(figsize=(6, 5))
    im = a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    a.set(title=f"Zero-shot on unseen test images ({test_acc:.1%})", xlabel="predicted", ylabel="true")
    a.set_xticks(range(10)); a.set_xticklabels(class_names, rotation=90, fontsize=8)
    a.set_yticks(range(10)); a.set_yticklabels(class_names, fontsize=8)
    fig.colorbar(im, ax=a); fig.tight_layout(); fig.savefig(f"{out}/clip_confusion.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    BATCH, STEPS = 128, 800
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train = datasets.CIFAR10("data", train=True, download=True, transform=tf)
    test = datasets.CIFAR10("data", train=False, download=True, transform=tf)
    class_names = train.classes
    tok = Tokenizer(class_names)

    # The ONE batch we overfit on
    images, labels = next(iter(DataLoader(train, batch_size=BATCH, shuffle=True)))
    ids = tok([tok.prompt(class_names[l]) for l in labels]).to(device)
    images, labels_d = images.to(device), labels.to(device)

    model = CLIP(tok.vocab_size, tok.ctx_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    @torch.no_grad()
    def sim_matrix():
        model.eval()
        txt = model.encode_text(tok([tok.prompt(c) for c in class_names]).to(device))
        return model.encode_image(images) @ txt.T

    sim_before = sim_matrix()
    hist = {"step": [], "loss": [], "acc": []}
    for step in range(STEPS + 1):
        model.train()
        loss = clip_loss(model(images, ids), labels_d)
        opt.zero_grad(); loss.backward(); opt.step()
        acc, _ = zero_shot_acc(model, tok, images, labels, class_names)
        hist["step"].append(step); hist["loss"].append(loss.item()); hist["acc"].append(acc)
        if step % 25 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}  zero-shot acc on train batch {acc:.2%}")

    # Results
    train_acc, _ = zero_shot_acc(model, tok, images, labels, class_names)
    timgs, tlabels = next(iter(DataLoader(test, batch_size=1000, shuffle=False)))
    test_acc, pred = zero_shot_acc(model, tok, timgs, tlabels, class_names)
    print(f"\ntrain batch ({BATCH} imgs) acc: {train_acc:.2%}")
    print(f"unseen test (1000 imgs) acc: {test_acc:.2%}   (chance = 10%)")
    print("classes in the batch:", sorted({class_names[l] for l in labels.tolist()}))
    print("test preds histogram:", torch.bincount(pred, minlength=10).tolist())

    cm = torch.zeros(10, 10)
    for t, p in zip(tlabels, pred):
        cm[t, p] += 1
    plot_results(hist, sim_before, sim_matrix(), labels, cm.numpy(), class_names, test_acc)
    print("plots saved to outputs/clip_*.png")




