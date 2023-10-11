import urllib
import io
import os
import zlib
import json

import redis

import torch
import timm
import numpy as np

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from fastapi import FastAPI, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from typing import Dict, Annotated

app = FastAPI(title="Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TIMM_MODEL = os.environ.get("TIMM_MODEL", "resnet18")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

@app.on_event("startup")
async def initialize():
    # initializes model, categories, redis connection
    global model
    print(f"loading model {TIMM_MODEL}...")
    model = timm.create_model(TIMM_MODEL, pretrained=True)
    model.eval()
    print(f"loaded model {TIMM_MODEL}")

    global categories
    # Download human-readable labels for ImageNet.
    # get the classnames
    url, filename = (
        "<https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt>",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    global redis_pool
    print(f"creating redis connection with {REDIS_HOST} {REDIS_PORT}")
    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0, decode_responses=True
    )

def get_redis():
    # Here, we re-use our connection pool
    # not creating a new one
    return redis.Redis(connection_pool=redis_pool)

def predict(inp_img: Image) -> Dict[str, float]:
    config = resolve_data_config({}, model=TIMM_MODEL)
    transform = create_transform(**config)

    img_tensor = transform(inp_img).unsqueeze(0)  # transform and add batch dimension

    # inference
    with torch.no_grad():
        out = model(img_tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        confidences = {categories[i]: float(probabilities[i]) for i in range(1000)}

    return confidences

async def write_to_cache(file: File, result: Dict[str, float]) -> None:
    cache = get_redis()

    hash = zlib.adler32(file)
    cache.set(hash, json.dumps(result))

@app.post("/infer")
async def infer(image: Annotated[bytes, File()]):
    img: Image.Image = Image.open(io.BytesIO(image))
    img = img.convert("RGB")

    predictions = predict(img)

    top10 = {
        k: v
        for k, v in list(sorted(predictions.items(), key=lambda x: x[1], reverse=True))[
            :10
        ]
    }

    await write_to_cache(image, top10)

    return top10

# uvicorn server:app --host 0.0.0.0 --port 8000 --reload
