[← All projects](../README.md)

# 4. CLIP from scratch

A tiny CLIP: an image encoder and a text encoder trained so matching pairs land close together. Then it classifies images with no classifier head, just by comparing them to captions.

| | |
| --- | --- |
| Code | [`scripts/clip.py`](../scripts/clip.py) |
| Task | Contrastive image-text training, then zero-shot classification |
| Data | One CIFAR-10 batch of 128 images (train), 1000 unseen images (test) |
| Model | 4-block CNN + 2-layer text transformer, 128-d shared space |
| Result | 100% zero-shot on the training batch, 34.3% on unseen images (chance 10%) |

## The question

How does CLIP classify images it was never trained to classify? I wanted to build the full pipeline small enough to see every step work, and the cheapest proof is overfitting one batch.

## Theory

**One shared space.** An image encoder maps a photo to a vector. A text encoder maps a caption to a vector of the same size. Both are L2-normalized, so the dot product is cosine similarity.

**Contrastive loss.** For a batch of N images and N captions, build the N x N similarity matrix. The right pairs are on the diagonal. Cross-entropy pushes each row (image to captions) and each column (caption to images) toward its correct match. That is the symmetric InfoNCE loss.

**Temperature.** Similarities are multiplied by a learnable scale before the softmax. A higher scale makes the softmax sharper.

**Zero-shot classification.** Write one caption per class ("a photo of a cat", "a photo of a dog", ...), embed them, and pick the class whose caption is closest to the image. No classifier head is trained.

```
image ─► CNN ─► vector ─┐
                        ├─► similarity matrix ─► symmetric cross-entropy
text  ─► Transformer ─► vector ─┘
Zero-shot: score the image against "a photo of a {class}" for all 10 classes, take the best.
```

**A twist with CIFAR-10.** A batch of 128 images has only 10 classes, so many captions are identical. Treating only the diagonal as correct would punish the model for matching a cat image to another cat's caption. So all same-class pairs count as positives (soft targets). That is also why the loss never reaches 0.

## How I built it

- Image encoder: 4 conv blocks, global average pool, linear to 128-d.
- Text encoder: word-level tokenizer over the prompt vocabulary, 2-layer pre-norm transformer (width 128, 4 heads), read out at the EOS token, linear to 128-d.
- Loss: symmetric soft-target InfoNCE with a learnable temperature.
- Training: AdamW lr 1e-3, 800 steps on one batch of 128 images.
- Plots: training curves, image-vs-caption similarity before and after, test confusion matrix.

## Run it

```bash
uv run python scripts/clip.py
```

## Results

Loss falls and flattens; zero-shot accuracy on the batch hits 100% early.

![CLIP training](../outputs/clip_training.png)

Image-vs-caption similarity. Before training it is noise. After, each image lights up on its own class.

![CLIP similarity](../outputs/clip_similarity.png)

On 1000 unseen images it gets 34.3%. Better than chance, but many mistakes.

![CLIP confusion](../outputs/clip_confusion.png)

## What I learned

- The pipeline works end to end: two encoders, shared space, contrastive loss, zero-shot by text.
- Overfitting one batch is a fast, honest check that the code is right before spending real compute.
- One batch memorizes; it does not generalize. Real CLIP used 400M pairs for a reason.

## Next

- Train on the full CIFAR-10 train set with mini-batches.
- Use a real caption dataset (Flickr8k / COCO) instead of templates.
- Swap in a pretrained image encoder and compare zero-shot accuracy.

## References

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020), 2021
