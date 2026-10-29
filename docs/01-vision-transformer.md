[← All projects](../README.md)

# 1. Vision Transformer

A small ViT written from scratch and trained on a slice of CIFAR-10. It reaches 47% test accuracy in 10 epochs and is still climbing when training stops.

| | |
| --- | --- |
| Code | [`scripts/vit.py`](../scripts/vit.py) |
| Task | Image classification, 10 classes |
| Data | CIFAR-10, 10k train / 2k test images, 32x32 |
| Model | `dim=128, depth=4, heads=4`, 4x4 patches |
| Hardware | Mac (MPS), about 3 minutes |
| Result | 47% test accuracy (chance is 10%) |

## The question

Transformers took over NLP by treating text as a sequence of tokens. Can the same block work on images if you just cut the image into pieces and call each piece a token? And how badly does it do with little data?

## Theory

**Patches as words.** A 32x32 image cut into 4x4 patches gives 64 patches. Each patch is flattened and projected to a vector. Now the image is a sequence of 64 tokens, just like a sentence.

**Position.** Self-attention does not know order. Shuffle the patches and it gives the same answer. So we add a learnable position embedding to each patch.

**The [CLS] token.** One extra learnable token is put at the front. It attends to every patch, and after the last block its output is the summary used for classification.

**Self-attention.** Every patch builds a query, a key and a value. Attention weights are `softmax(QKᵀ / √d)`, and each patch's output is a weighted mix of all values. A patch in the corner can look at a patch in the middle in the very first layer. A CNN needs many layers to see that far.

**Why ViTs need data.** CNNs have built-in assumptions (nearby pixels matter, a cat is a cat wherever it sits). ViTs have almost none and must learn them from data. The original paper needed hundreds of millions of images to beat CNNs. 10k images is tiny, so I expected a modest number.

```
32x32 image ─► 64 patches (4x4) ─► linear ─► [CLS] + patches + position ─► 4 x Transformer block ─► CLS ─► 10 classes
```

## How I built it

- `PatchEmbed`: one `Conv2d` with kernel = stride = 4. It cuts patches and projects them in one go.
- `Attention`: multi-head self-attention with one fused QKV linear layer.
- `Block`: pre-norm. LayerNorm, attention, residual, then LayerNorm, MLP (4x width), residual.
- `ViT`: learnable `[CLS]` token and position embeddings. The CLS output goes to a linear classifier.
- Training: AdamW (lr 1e-3, weight decay 0.05), cosine LR, batch 128, 10 epochs.

## Run it

```bash
uv run python scripts/vit.py
```

The plot is saved to `outputs/vit_training.png`.

## Results

![ViT training](../outputs/vit_training.png)

- Accuracy rises every epoch and ends at 47%. The curve has not flattened, so more epochs would still help.
- That is well above chance (10%) but still modest. Expected, given how little data the ViT had. (I have not run a CNN on the same 10k split yet; that is the fair comparison.)

## What I learned

- The whole architecture is short. Patch embed, attention, MLP, residuals. Most of the work is in the shapes.
- One strided conv is the cleanest way to cut and embed patches.
- Data size decides whether a ViT is a good choice. On small data, a CNN or a pretrained ViT wins.

## Next

- Train on all 50k images and for more epochs.
- Add stronger augmentation (RandAugment, Mixup, CutMix), as DeiT does.
- Fine-tune a pretrained ViT and compare.
- Plot attention maps from the CLS token.

## References

- Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), 2020
- Touvron et al., [DeiT: Training data-efficient image transformers](https://arxiv.org/abs/2012.12877), 2020
