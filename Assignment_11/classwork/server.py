import requests
import torch
import io

import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load dataset
dataset = load_dataset("jxie/flickr8k")

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# load image embeddings
img_embeds = torch.tensor(np.load("image_embeds.npy"))

@app.get("/text-to-image")
async def find_image(text: str):
    inputs = processor([text], padding=True, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    image_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    img_idx = torch.argmax(logits_per_text[0]).item()
    similar_img = dataset['train'][img_idx]['image']

    img_byte_arr = io.BytesIO()
    similar_img.save(img_byte_arr, format="JPEG")

    return Response(img_byte_arr.getvalue(), media_type="image/jpeg")

@app.get("/health")
async def health():
    return {"message": "ok"}
