"""
Minimal LLaVA-style multimodal language model, trained as an image captioner on Flickr8k.

    image ──► Vision Encoder (CLIP ViT) ──► Projector (MLP) ──► image tokens ─┐
                                                                              ├─► [img | text] ──► Causal LM ──► logits
    text  ──► Tokenizer ──► Token Embedding ──► text tokens ──────────────────┘

Three parts:
  1. Vision encoder : turns an image into a sequence of patch features (frozen).
  2. Projector      : maps vision features into the LLM's embedding space, so an
                      image patch looks like "just another word embedding". Only this is trained.
  3. Language model : decoder-only transformer with causal attention (frozen). It reads the
                      image tokens as a prefix, then predicts the caption one token at a time.

Because the vision encoder is frozen, its patch features are computed once and cached.

    uv run python scripts/multimodal_llm.py --overfit      # sanity: one batch, loss -> ~0
    uv run python scripts/multimodal_llm.py                # full run
"""
import argparse
import math
import os
import re
import time
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

CLIP_PATH = "models/clip-vit-base-patch32"
LLM_PATH = "models/qwen2.5-0.5b-instruct"
DATA = "data/flickr8k/data"
CACHE = "data/flickr8k"
OUT = "outputs"
MAX_LEN = 40

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)


class Projector(nn.Module):
    def __init__(self, d_vis=768, d_llm=896):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_vis, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))

    def forward(self, x):   # [B,49,768] -> [B,49,896]
        return self.net(x)


class MultiModalLLM(nn.Module):
    def __init__(self, clip_path, llm_path):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(clip_path)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch.float32)

        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False

        self.projector = Projector(self.vision.config.hidden_size, self.llm.config.hidden_size)

    @torch.no_grad()
    def patch_features(self, pixel_values):
        return self.vision(pixel_values=pixel_values).last_hidden_state[:, 1:]  # drop CLS -> [B,49,768]

    def encode_image(self, feats):
        return self.projector(feats)                             # [B,49,896]

    def forward(self, feats, input_ids, attention_mask):
        img = self.encode_image(feats)                           # [B,N,D]
        txt = self.llm.get_input_embeddings()(input_ids)         # [B,T,D]
        x = torch.cat([img, txt], dim=1)                         # image is the prefix

        B, N = img.shape[:2]
        img_mask = torch.ones(B, N, dtype=attention_mask.dtype, device=attention_mask.device)
        mask = torch.cat([img_mask, attention_mask], dim=1)      # [B,N+T]

        img_labels = torch.full((B, N), -100, dtype=input_ids.dtype, device=input_ids.device)
        txt_labels = input_ids.masked_fill(attention_mask == 0, -100)
        labels = torch.cat([img_labels, txt_labels], dim=1)      # [B,N+T]

        return self.llm(inputs_embeds=x, attention_mask=mask, labels=labels)

    @torch.no_grad()
    def generate(self, feats, eos_id, max_new_tokens=30, **kw):
        img = self.encode_image(feats)
        mask = torch.ones(img.shape[:2], dtype=torch.long, device=img.device)
        return self.llm.generate(
            inputs_embeds=img,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            **kw,
        )


# ---------- data ----------
def load_data():
    ds = load_dataset("parquet", data_files={s: f"{DATA}/{s}-*.parquet" for s in ["train", "validation", "test"]})
    caps = {s: [[r[f"caption_{i}"].strip().rstrip(" .").strip() for i in range(5)] for r in ds[s]] for s in ds}
    return ds, caps


@torch.no_grad()
def get_feats(model, ds, split, processor):
    path = f"{CACHE}/clip_feats_{split}.pt"
    if os.path.exists(path):
        return torch.load(path)
    out = []
    for i in range(0, len(ds[split]), 64):
        imgs = [im.convert("RGB") for im in ds[split][i:i + 64]["image"]]
        px = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
        out.append(model.patch_features(px).half().cpu())
        print(f"  cache {split} {i + len(imgs)}/{len(ds[split])}", end="\r")
    feats = torch.cat(out)
    torch.save(feats, path)
    return feats


def make_batch(tok, caps, feats, idx, cap_ids):
    text = [caps[i][c] + tok.eos_token for i, c in zip(idx, cap_ids)]
    t = tok(text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return feats[idx].float().to(device), t.input_ids.to(device), t.attention_mask.to(device)


# ---------- eval ----------
@torch.no_grad()
def val_loss(model, tok, caps, feats, shuffle=False, bs=50):
    model.eval()
    n, total, count = len(feats), 0.0, 0
    if shuffle:   # captions keep their order, images are permuted -> mismatched pairs
        feats = feats[torch.randperm(n, generator=torch.Generator().manual_seed(1))]
    for i in range(0, n, bs):
        idx = list(range(i, min(i + bs, n)))
        f, ids, m = make_batch(tok, caps, feats, idx, [0] * len(idx))
        loss = model(f, ids, m).loss
        k = int(m.sum())
        total += loss.item() * k
        count += k
    model.train()
    return total / count


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def corpus_bleu4(hyps, refs):
    """Corpus BLEU-4 with brevity penalty; refs = list of lists of reference strings."""
    match, total = [0] * 4, [0] * 4
    hl = rl = 0
    for h, rs in zip(hyps, refs):
        h = words(h)
        rs = [words(r) for r in rs]
        hl += len(h)
        rl += min((abs(len(r) - len(h)), len(r)) for r in rs)[1]
        for n in range(1, 5):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            mx = Counter()
            for r in rs:
                for g, c in Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1)).items():
                    mx[g] = max(mx[g], c)
            match[n - 1] += sum(min(c, mx[g]) for g, c in hc.items())
            total[n - 1] += sum(hc.values())
    if min(match) == 0:
        return 0.0
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    return bp * math.exp(sum(math.log(m / t) for m, t in zip(match, total)) / 4)


@torch.no_grad()
def caption(model, tok, feats, bs=50, **kw):
    model.eval()
    out = []
    for i in range(0, len(feats), bs):
        g = model.generate(feats[i:i + bs].float().to(device), tok.eos_token_id, **kw)
        out += [s.strip() for s in tok.batch_decode(g, skip_special_tokens=True)]
    model.train()
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ds, caps = load_data()
    tok = AutoTokenizer.from_pretrained(LLM_PATH)
    tok.padding_side = "right"
    processor = CLIPImageProcessor.from_pretrained(CLIP_PATH)
    model = MultiModalLLM(CLIP_PATH, LLM_PATH).to(device)
    feats = {s: get_feats(model, ds, s, processor) for s in ["train", "validation", "test"]}
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device}  trainable params={n_train / 1e6:.2f}M  train={len(feats['train'])} val={len(feats['validation'])}")

    params = list(model.projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    if args.overfit:   # sanity: one fixed batch, loss should go to ~0
        idx = list(range(args.bs))
        f, ids, m = make_batch(tok, caps["train"], feats["train"], idx, [0] * len(idx))
        for step in range(100):
            loss = model(f, ids, m).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 10 == 0 or step == 99:
                print(f"overfit step {step:3d}  loss {loss.item():.4f}")
        return

    n = len(feats["train"])
    steps_per_epoch = n // args.bs
    total_steps = steps_per_epoch * args.epochs
    warmup = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))
    )

    train_loss, lrs, gnorms, val_pts = [], [], [], []
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n).tolist()
        for b in range(steps_per_epoch):
            idx = perm[b * args.bs:(b + 1) * args.bs]
            cap_ids = torch.randint(0, 5, (len(idx),)).tolist()   # random one of 5 captions per image
            f, ids, m = make_batch(tok, caps["train"], feats["train"], idx, cap_ids)
            loss = model(f, ids, m).loss
            opt.zero_grad()
            loss.backward()
            gnorms.append(nn.utils.clip_grad_norm_(params, 1.0).item())
            opt.step()
            sched.step()
            train_loss.append(loss.item())
            lrs.append(sched.get_last_lr()[0])
            step += 1
            if step % 25 == 0:
                print(f"ep {ep} step {step}/{total_steps}  loss {loss.item():.3f}  gnorm {gnorms[-1]:.2f}  {time.time() - t0:.0f}s")
            if step % 100 == 0 or step == total_steps:
                v = val_loss(model, tok, caps["validation"], feats["validation"])
                val_pts.append((step, v))
                print(f"  val loss {v:.3f}  ppl {math.exp(v):.1f}")

    torch.save(model.projector.state_dict(), f"{OUT}/projector.pt")

    # --- ablation: does the model use the image? ---
    v_ok = val_pts[-1][1]
    v_shuf = val_loss(model, tok, caps["validation"], feats["validation"], shuffle=True)
    v_zero = val_loss(model, tok, caps["validation"], torch.zeros_like(feats["validation"]))
    print(f"val loss: correct image {v_ok:.3f} | shuffled image {v_shuf:.3f} | zero image {v_zero:.3f}")

    # --- caption quality on test ---
    gen_kw = dict(num_beams=3, repetition_penalty=1.2)
    hyps = caption(model, tok, feats["test"], **gen_kw)
    refs = caps["test"]
    bleu = corpus_bleu4(hyps, refs)
    print(f"test BLEU-4 {bleu:.3f}  distinct captions {len(set(hyps))}/{len(hyps)}")

    # --- plots ---
    k = 20
    smooth = [sum(train_loss[max(0, i - k):i + 1]) / len(train_loss[max(0, i - k):i + 1]) for i in range(len(train_loss))]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(train_loss, alpha=0.25, label="train (raw)")
    ax[0].plot(smooth, label="train (smoothed)")
    ax[0].plot(*zip(*val_pts), "o-", label="val")
    ax[0].set(xlabel="step", ylabel="loss", title="Loss"), ax[0].legend()
    ax[1].plot(lrs), ax[1].set(xlabel="step", title="Learning rate")
    ax[2].plot(gnorms), ax[2].set(xlabel="step", title="Grad norm (pre-clip)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/mllm_loss.png", dpi=120)
    plt.close()

    sel = list(range(0, 12))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    sample_hyps = caption(model, tok, feats["validation"][sel], **gen_kw)
    for a, i, h in zip(axes.flat, sel, sample_hyps):
        a.imshow(ds["validation"][i]["image"].convert("RGB"))
        a.axis("off")
        a.set_title(f"PRED: {h}\nGT: {caps['validation'][i][0]}", fontsize=8, wrap=True)
    plt.tight_layout()
    plt.savefig(f"{OUT}/mllm_samples.png", dpi=120)
    plt.close()
    print("saved outputs/mllm_loss.png, outputs/mllm_samples.png, outputs/projector.pt")


if __name__ == "__main__":
    main()
