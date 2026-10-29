[← All projects](../README.md)

# 3. Transfer learning

An ImageNet-pretrained ResNet-18 taught to recognise 102 kinds of flowers. Two stages: train the new head first, then fine-tune everything.

| | |
| --- | --- |
| Code | [`scripts/cnn.py`](../scripts/cnn.py) |
| Task | Image classification, 102 classes |
| Data | Oxford Flowers-102, 224x224 crops |
| Model | torchvision ResNet-18, ImageNet weights, new 102-way head |
| Result | Confusion matrix is close to a clean diagonal; best checkpoint in `outputs/best.pt` |

## The question

Flowers-102 has only about 10 training images per class. Training a CNN from scratch on that would overfit badly. How far can you get by reusing what a network already learned on ImageNet?

## Theory

**What a pretrained net knows.** Early layers detect edges and colors. Middle layers detect textures and parts. Late layers combine parts into objects. The early and middle layers are useful for almost any photo, flowers included. Only the final classifier is specific to ImageNet's 1000 classes.

**Two-stage fine-tuning.**
1. Freeze the backbone and train only the new head. The head starts random, and its large early gradients would wreck the pretrained weights if they could reach them.
2. Unfreeze everything and train with a small learning rate for the backbone and a larger one for the head. Now the features can adapt to flowers without being thrown away.

**Normalization.** Inputs use ImageNet's mean and std, because that is what the backbone saw during pretraining.

```
pretrained ResNet-18 ─► swap last layer for 102 classes
  stage 1: freeze backbone, train head only (5 epochs)
  stage 2: unfreeze all, cosine LR (15 epochs)
```

## How I built it

- Train augmentation: `RandomResizedCrop(224)` and horizontal flip. Eval: resize 256, center crop 224.
- Stage 1: Adam on `model.fc` only, lr 1e-3, 5 epochs.
- Stage 2: Adam with two parameter groups, backbone lr 1e-4 and head lr 1e-3, cosine schedule, 15 epochs.
- Keep the best val checkpoint, then draw a confusion matrix on the test set with scikit-learn.

## Run it

```bash
uv add scikit-learn
uv run python scripts/cnn.py
```

## Results

![Flowers-102 confusion matrix](../outputs/confusion_matrix.png)

- The diagonal is strong across almost all 102 classes, so most flowers are classified correctly.
- The off-diagonal spots are the classes worth looking at by eye. With 102 classes, a few lookalike pairs are expected.

## What I learned

- With tiny datasets, pretrained weights do most of the work.
- Freezing first, then unfreezing with a lower backbone LR, is a simple recipe that avoids wrecking the features.
- A confusion matrix tells you more than a single accuracy number, especially with 102 classes.

## Next

- Record final test accuracy in the log and in this page.
- Compare ResNet-18 with ResNet-50 and a pretrained ViT.
- Try linear probing only (stage 1) vs full fine-tuning to see how much stage 2 adds.

## References

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015
- Yosinski et al., [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792), 2014
