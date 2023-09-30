from typing import Annotated
import requests
import io

import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    img = Image.open(io.BytesIO(file))
    img = img.convert("RGB")

    img_np = np.array(img)

    print(f"shape = {img_np.shape}")

    ret_img = io.BytesIO()
    img.save(ret_img, format="jpeg")
    return Response(ret_img.getvalue(), media_type='image/jpeg')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    print(f"filename = {file.filename=}")

    file_b = await file.read()

    img = Image.open(io.BytesIO(file_b))
    img = img.convert("RGB")

    img_np = np.array(img)

    print(f"shape = {img_np.shape}")

    ret_img = io.BytesIO()
    img.save(ret_img, format="jpeg")
    return Response(ret_img.getvalue(), media_type='image/jpeg')

@app.get("/health")
async def health():
    return {"message": "ok"}
