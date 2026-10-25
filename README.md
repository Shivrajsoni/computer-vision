# Computer Vision from scratch

My notes and code for learning deep learning in computer vision. Every project is one small file, with a picture of what it does and the result I got.

**Run anything from the repo root:** `uv run python scripts/<file>.py`
(Setup once: `uv sync`. Scripts use CUDA, then Apple MPS, then CPU.)

## At a glance

| # | Project | File | One-line result |
| --- | --- | --- | --- |
| 1 | Vision Transformer | [`scripts/vit.py`](scripts/vit.py) | 47% test accuracy on CIFAR-10 after 10 epochs on 10k images |
| 2 | ResNet internals | [`notebooks/resnet.ipynb`](notebooks/resnet.ipynb) | Blocks built by hand, Conv+BN fusion for faster inference |
| 3 | Transfer learning | [`scripts/cnn.py`](scripts/cnn.py) | Pretrained ResNet-18 fine-tuned on Flowers-102, strong diagonal confusion matrix |
| 4 | CLIP from scratch | [`scripts/clip.py`](scripts/clip.py) | 100% zero-shot on the one batch it memorized, 34% on unseen images |
| 5 | Multimodal LLM | [`scripts/multimodal_llm.py`](scripts/multimodal_llm.py) | Flickr8k captioner, only a 1.5M projector trained: val loss 2.68 (ppl 14.5), captions get the gist but hallucinate details |

---

## 1. Vision Transformer (`vit.py`)

**Idea:** cut the image into patches, treat each patch like a word, and run a transformer.

```
32x32 image ─► 64 patches (4x4) ─► linear ─► [CLS] + patches + position ─► 4 x Transformer block ─► CLS ─► 10 classes
```

**How I built it**
- `PatchEmbed`: one strided `Conv2d` cuts patches and projects each to a vector.
- `Attention`: multi-head self-attention with one fused QKV layer.
- `Block`: LayerNorm, attention, LayerNorm, MLP, each with a residual.
- `ViT`: learnable `[CLS]` token and position embeddings; the CLS output feeds the classifier.
- Training: 10k train / 2k test CIFAR-10 images, `dim=128, depth=4, heads=4`, AdamW, cosine LR, 10 epochs (~3 min on a Mac).

**Result:** accuracy climbs steadily and is still rising at epoch 10. A ViT needs lots of data; 10k images is small.

![ViT training](outputs/vit_training.png)

---

## 2. ResNet internals (`notebooks/resnet.ipynb`)

**Idea:** a layer learns `x + f(x)` instead of `f(x)`. The skip connection lets very deep nets train.

```
x ─┬─► conv ─► BN ─► ReLU ─► conv ─► BN ─┐
   └──────────── (1x1 conv if shape changes) ─┴─► + ─► ReLU
```

**How I built it**
- Batched inference with a pretrained ResNet-50 (`inference_mode`, fp16).
- `BasicBlock` (ResNet-18/34) and `Bottleneck` (ResNet-50+).
- `make_stage` stacks blocks; only the first block downsamples.
- Conv+BatchNorm fusion by hand (`manual_fuse_conv_bn`) and with `fuse_conv_bn_eval`. Same outputs, fewer ops at inference.

---

## 3. Transfer learning (`cnn.py`)

**Idea:** reuse an ImageNet-pretrained ResNet-18 and teach it 102 flower types.

```
pretrained ResNet-18 ─► swap last layer for 102 classes
  stage 1: freeze backbone, train head only (5 epochs)
  stage 2: unfreeze all, cosine LR (15 epochs)
```

**Result:** the confusion matrix is almost a perfect diagonal, so most flowers are classified correctly. The best checkpoint is saved to `outputs/best.pt`.

![Flowers-102 confusion matrix](outputs/confusion_matrix.png)

Needs scikit-learn: `uv add scikit-learn`.

---

## 4. CLIP from scratch (`clip.py`)

**Idea:** put images and captions in the same space so a photo and "a photo of a cat" land close together. Then classify with **no classifier head**: pick the class whose caption is nearest.

```
image ─► CNN ─► vector ─┐
                        ├─► similarity matrix ─► symmetric cross-entropy
text  ─► Transformer ─► vector ─┘
Zero-shot: score the image against "a photo of a {class}" for all 10 classes, take the best.
```

**How I built it**
- Image encoder: 4-block CNN. Text encoder: 2-layer transformer over a tiny word-level vocabulary.
- Both outputs are L2-normalized (128-d) and scaled by a learnable temperature.
- A batch has only 10 classes, so same-class pairs all count as positives (soft targets, not the identity matrix). That is why the loss never reaches 0.
- Overfit ONE CIFAR-10 batch (128 images, 800 steps) to prove the whole pipeline works.

**Result:** 100% zero-shot accuracy on the training batch, 34.3% on 1000 unseen images (chance is 10%). It memorizes; it does not generalize. That is expected from one batch.

Loss falls and flattens; accuracy hits 100% early:

![CLIP training](outputs/clip_training.png)

Image-vs-caption similarity. Before training: noise. After: each image lights up on its own class.

![CLIP similarity](outputs/clip_similarity.png)

Unseen test images: better than chance, but many mistakes.

![CLIP confusion](outputs/clip_confusion.png)

---

## 5. Multimodal LLM (`multimodal_llm.py`)

**Idea:** turn image patches into "words" a language model can read, then let it write text. This is the LLaVA recipe, stage 1 (alignment): train only the bridge between a frozen vision model and a frozen LLM.

```
image ─► CLIP ViT (frozen) ─► MLP projector (trained) ─► 49 image tokens ─┐
                                                                          ├─► Qwen2.5-0.5B (frozen) ─► next word
caption ─► tokenizer ─► embedding table ──────────────────────────────────┘
```

**Why a projector?** CLIP's features (768-d) live in CLIP's space. Qwen reads vectors in its own 896-d embedding space, and the axes of the two are unrelated. The projector learns the map from one to the other. Its outputs are "soft tokens": they do not match real words, they are whatever makes the caption loss go down.

**What one training example looks like** (image + the caption "a dog runs on grass"):

```
input   : [I1 ... I49]  a    dog   runs  on    grass  EOS      (49 image vectors + 6 text vectors)
predict :  ...    I49   a    dog   runs  on    grass  EOS
                   │    │     │     │    │      │      │
target  :          a   dog  runs   on  grass   EOS     -
```

- Each position predicts the **next** token. Loss = cross-entropy against the true token **IDs** (not embeddings).
- Image positions and padding are masked with -100, so only caption tokens count.
- Teacher forcing: training feeds the true previous words, so all predictions come from one forward pass. `generate` feeds back its own words one at a time.
- Gradients flow back through frozen Qwen into the projector, the only thing that updates.
- Each image has 5 captions; every step picks one at random. The model learns what they share ("dog", "grass", "running"), and this is also why the loss cannot reach 0.

**How I built it**
- Frozen CLIP ViT-B/32 (49 patch tokens, CLS dropped) and frozen Qwen2.5-0.5B-Instruct. Only the 2-layer MLP projector (768 -> 896 -> 896, **1.49M params**) is trained.
- Vision is frozen, so CLIP features are computed once and cached to disk. Training reads the cache.
- Flickr8k: 6000 train / 1000 val / 1000 test images, 5 captions each.
- AdamW, lr 1e-3, 100 warmup steps then cosine decay, grad clip 1.0, batch 32, 3 epochs (561 steps). About 15 min on a Colab T4.
- Generation: beam search (3 beams) with repetition penalty 1.2.
- Evaluation: val loss, a shuffled-image and a zero-image ablation, test BLEU-4 (own implementation in the script), and a grid of sample captions.

**Run it**
- Colab (recommended, T4): upload [`notebooks/multimodal_llm_colab.ipynb`](notebooks/multimodal_llm_colab.ipynb), set the runtime to T4 GPU, run all. It downloads the models and data and saves plots and `projector.pt`.
- Local: `uv run python scripts/multimodal_llm.py --overfit` (sanity check, loss should fall toward 0), then `uv run python scripts/multimodal_llm.py`. On a Mac with MPS this is much slower than a T4.

### Results

Training log from the Colab run:

![MLLM training log](outputs/mllm_training_log.png)

| Check | Value | Meaning |
| --- | --- | --- |
| Val loss, correct image | **2.675** (perplexity 14.5) | on average the model is about as unsure as picking among 14 words per step |
| Val loss, shuffled images | 3.175 | images swapped between captions: worse, so the image is used |
| Val loss, zero image | above 6 (log cut off at 6.5) | no image: far worse |
| Test BLEU-4 | not recorded | cut off in the log; rerun to get it |
| Speed | ~1.6 s/step on T4 | 561 steps in about 15 min |

![MLLM loss](outputs/mllm_loss.png)

How to read the three plots (loss, learning rate, grad norm):
- **Loss:** drops from about 8 to 3 in the first 100 steps (the model learns the caption style and uses the image), then creeps from 3.2 to 2.68. Train and val track each other, so there is **no overfitting**. The model is limited by capacity, since only the projector learns.
- **Learning rate:** linear warmup to 1e-3 over 100 steps, then cosine decay to 0.
- **Grad norm (before clipping):** huge at the start (up to 450) while the projector is random, then settles near 1. Warmup and clipping kept training stable.

Sample captions on validation images (PRED is the model, GT is one human caption):

![MLLM samples](outputs/mllm_samples.png)

- **Gets the gist:** "Two dogs are playing in the grass", "A group of young girls are playing on a grassy field".
- **Misses details and actions:** a toddler skateboarding in a blue helmet is "a boy in a red shirt running down the street"; a baseball slide is a football kick; ice skaters are "walking along a street"; a boy in a bubble bath is "playing with a toy". Colors and counts are often wrong.
- **Generic wording:** many captions start "A group of people are...". This is typical when only the projector is trained and the LLM stays frozen.

**What this tells me:** the pipeline works (the image really is used, shown by the ablations), but a frozen LLM that must read soft tokens it never saw gives coarse captions. The 0.5-nat gap between correct and shuffled images shows much of the loss is still the language prior.

**Next steps (stage 2)**
1. LoRA on Qwen's attention layers so the LLM itself adapts to image tokens.
2. Instruction data in Qwen's chat format (`user: [image] question -> assistant: answer`), with the prompt masked out of the loss.
3. Compute logits only on answer positions (saves memory and time).
4. Try CLIP's penultimate-layer features, as LLaVA does, instead of the last layer.
5. Check object hallucination with yes/no questions ("Is there a dog?").

---

## Repo layout

```
scripts/     vit.py  cnn.py  clip.py  multimodal_llm.py
notebooks/   resnet.ipynb  multimodal_llm_colab.ipynb
outputs/     plots (committed) and checkpoints (git-ignored)
data/        datasets (git-ignored)
```

## References

- He et al., [Deep Residual Learning](https://arxiv.org/abs/1512.03385), 2015
- Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), 2020
- Radford et al., [CLIP](https://arxiv.org/abs/2103.00020), 2021
- Liu et al., [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485), 2023
