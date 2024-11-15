import nest_asyncio
import uvicorn
import yaml
import os
from src.utils import byte2Numpy, numpy2base64, resize_with_aspect_ratio, make_palette
from src.analyzer import Analyzer
from fastapi import FastAPI, Request, File, UploadFile


app = FastAPI()
config = yaml.safe_load(open("config.yaml", "r"))
model_path = config["model_path"]
port = int(os.getenv("PORT", config["port"]))
default_size = config["default_image_size"]
device = os.getenv("DEVICE", config["device"])

analyzer = Analyzer(model_path, device=device)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/")
async def inference(request: Request, file: UploadFile = File(...)):

    byte_image = await file.read()
    origin = byte2Numpy(byte_image)
    resized_image = resize_with_aspect_ratio(origin, default_size)
    dominant_colors, optimal_k, _ = analyzer.process(resized_image)

    result = make_palette(resized_image, dominant_colors, optimal_k)
    data = {"colors": dominant_colors.tolist(), "image": numpy2base64(result)}

    return data


nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=port)
