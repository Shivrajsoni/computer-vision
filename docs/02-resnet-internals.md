[← All projects](../README.md)

# 2. ResNet internals

Residual blocks built by hand, stacked into stages, then Conv + BatchNorm fused for faster inference. No training here; this one is about understanding the parts.

| | |
| --- | --- |
| Code | [`notebooks/resnet.ipynb`](../notebooks/resnet.ipynb) |
| Topics | Batched inference, `BasicBlock`, `Bottleneck`, stages, Conv+BN fusion |
| Model | torchvision ResNet-50 (pretrained) plus my own blocks |

## The question

Why can a ResNet be 50 or 150 layers deep when older nets stopped improving after about 20? And what actually happens inside a block?

## Theory

**The degradation problem.** Before ResNet, adding layers to a plain CNN made training error go *up*. This was not overfitting. The deeper net was simply harder to optimize.

**The residual idea.** Instead of learning `H(x)`, a block learns `F(x) = H(x) - x` and outputs `x + F(x)`. If the best thing a block can do is nothing, it only has to push `F` to zero, which is easy. Gradients also flow straight back through the `+ x` path, so early layers still learn.

```
x ─┬─► conv ─► BN ─► ReLU ─► conv ─► BN ─┐
   └──────────── (1x1 conv if shape changes) ─┴─► + ─► ReLU
```

**Two block types.**
- `BasicBlock` (ResNet-18/34): two 3x3 convs. Output channels equal the base channels.
- `Bottleneck` (ResNet-50+): 1x1 to shrink channels, 3x3 to do the work, 1x1 to expand by 4x. Cheaper per layer, so the net can go deeper.

When the shape changes (stride 2, or more channels), the shortcut needs a 1x1 conv so `x` and `F(x)` can be added.

**Stages.** A ResNet is a stem plus 4 stages. Only the first block of a stage downsamples; the rest keep the shape.

**Conv + BN fusion.** At inference BatchNorm uses fixed running stats, so it is just a per-channel scale and shift. A scale and shift right after a conv can be folded into the conv's weights and bias:

```
W' = W · γ / √(var + ε)
b' = (b − mean) · γ / √(var + ε) + β
```

Same output, one op instead of two.

## How I built it

1. Batched inference with a pretrained ResNet-50: stack 8 requests into one tensor, `torch.inference_mode()`, fp16.
2. `BasicBlock` and `Bottleneck` from scratch, with the 1x1 projection shortcut when needed.
3. `make_stage` to stack blocks; I checked output shapes for a ResNet-18 stage and a ResNet-50 stage.
4. `manual_fuse_conv_bn` that applies the formula above, checked against PyTorch's `fuse_conv_bn_eval`.
5. `fuse_resnet_block` to fuse every Conv+BN pair in a block in place.

## Run it

```bash
uv run --with jupyter jupyter lab notebooks/resnet.ipynb
```

## Results

- The notebook feeds a `2x64x56x56` tensor (what comes out of the stem) through a ResNet-18 stage 2 and a ResNet-50 stage, and prints the output shapes to check downsampling and channel expansion.
- `manual_fuse_conv_bn` is compared against PyTorch's `fuse_conv_bn_eval`. Both should match the unfused conv + BN output up to float error.

## What I learned

- The skip connection is the whole trick. The rest is ordinary convs and BN.
- Fusion only works in eval mode. In training mode BN uses batch stats, which change every step.
- Bottlenecks are why ResNet-50 is not much slower than ResNet-34 despite being deeper.

## Next

- Time fused vs unfused ResNet-50 on MPS and CUDA.
- Try `torch.compile` and see if it fuses on its own.
- Build a full ResNet-18 from my blocks and load torchvision weights into it.

## References

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015
