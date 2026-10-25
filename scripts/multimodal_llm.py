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
from sympy import N
from regex import B
from numpy import dtype
from numpy import imag
import torch
import torch.nn as nn 

# Loading the Pretrained ViT Encoder

from transformers import CLIPVisionModel , CLIPImageProcessor

path = "models/clip-vit-base-patch32"
vision = CLIPVisionModel.from_pretrained(path).eval()

for p in vision.parameters():
    p.requires_grad = False 

processor = CLIPImageProcessor.from_pretrained(path)

px = processor(images="/Users/shivraj/Developer/computer-vision/data/flowers-102/jpg/image_00001.jpg", return_tensors="pt").pixel_values
feats = vision(pixel_values=px).last_hidden_state

# print(feats.shape)


# LLM Loading 

from transformers import AutoModelForCausalLM , AutoTokenizer

path = "models/qwen2.5-0.5b-instruct"
tok = AutoTokenizer.from_pretrained(path)

llm = AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32)

for p in llm.parameters():
    p.requires_grad = False

d_llm = llm.config.hidden_size  # 896 , the projector output size 
embed = llm.get_input_embeddings() # token ID -> vector_layer 

# Writing the Projector Layer 
from datasets import load_dataset

ds = load_dataset("parquet", data_files={
    "train": "data/flickr8k/data/train-*.parquet",
    "validation": "data/flickr8k/data/validation-*.parquet",
})



projector = nn.Sequential(
    nn.Linear(768,896),
    nn.GELU(),
    nn.Linear(896,896)
)

class Projector(nn.Module):
    def __init__(self, d_vis=768,d_llm=896):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_vis,d_llm) , nn.GELU() , nn.Linear(d_llm , d_llm))

    
    def forward(self,x):   # [B,49,768] -> [B,49,896]
        return self.net(x)


class MultiModalLLM(nn.Module):
    def __init__(self, clip_path, llm_path):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(clip_path)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path , dtype = torch.float32)

        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False
        
        self.projector = Projector(self.vision.config.hidden_size , self.llm.config.hidden_size)

    def encode_image(self, pixel_values):
        with torch.no_grad():
            feats = self.vision(pixel_values = pixel_values).last_hidden_state
        return self.projector(feats[:,1:])  # # drop CLS -> [B,49,896]

    def forward(self, pixel_values , input_ids , attention_mask):
        img = self.encode_image(pixel_values)                    # [B,N,D]
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
    def generate(self, pixel_values , max_new_tokens = 30):
        img = self.encode_image(pixel_values=pixel_values)
        mask = torch.ones(img.shape[:2] , dtype=torch.long , device=img.device)
        return self.llm.generate(
            input_embeds = img,
            attention_mask=mask , 
            max_new_tokens=max_new_tokens,
            pad_token_id = self.llm.config.eos_token_id
        )


model = MultiModalLLM("models/clip-vit-base-patch32", "models/qwen2.5-0.5b-instruct").to("mps") 
tok = AutoTokenizer.from_pretrained("models/qwen2.5-0.5b-instruct")   # add the AutoTokenizer import

t = tok(["a dog runs on grass" + tok.eos_token], return_tensors="pt").to("mps")
px = torch.randn(1, 3, 224, 224, device="mps")
out = model(px, t.input_ids, t.attention_mask)
print(out.logits.shape, out.loss)    # expect [1, 49+T, vocab], loss around 10+
print(sum(p.numel() for p in model.parameters() if p.requires_grad))   # ~1.4M
