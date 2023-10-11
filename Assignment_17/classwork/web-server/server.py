import os
import zlib
import json
import requests

import redis
import httpx

from fastapi import FastAPI, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from typing import Annotated

app = FastAPI(title="Web Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "<http://localhost:8000>")

@app.on_event("startup")
async def initialize():
    global redis_pool
    print(f"creating redis connection with {REDIS_HOST} {REDIS_PORT}")
    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0, decode_responses=True
    )

def get_redis():
    return redis.Redis(connection_pool=redis_pool)

async def check_cached(image: bytes):
    hash = zlib.adler32(image)
    cache = get_redis()

    data = cache.get(hash)

    return json.loads(data) if data else None

@app.post("/classify-imagenet")
async def classify_imagenet(image: Annotated[bytes, File()]):
    infer_cache = await check_cached(image)

    if infer_cache == None:
        async with httpx.AsyncClient() as client:
            try:
                # print("this")
                # response = await client.post(
                #     f"{MODEL_SERVER_URL}/infer", files={"image": image}
                # )

                # print(f"response", response)

                # return response.json()
                url = f"{MODEL_SERVER_URL}/infer"
                files = {"image": image}

                response = requests.post(url, files=files)

                response.raise_for_status()  # Raise an exception for HTTP errors (4xx and 5xx)

                return response.json()
            except Exception as e:
                print(f"ERROR :: {e}")
                raise HTTPException(status_code=500, detail="Error from Model Endpoint")

    return infer_cache

# uvicorn server:app --host 0.0.0.0 --port 9000 --reload
