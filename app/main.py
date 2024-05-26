import os

from dotenv import load_dotenv
from fastapi import Request
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from routers.predict_router import router2, router1
import mysql.connector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(BASE_DIR, 'model'))
DATA_DIR = os.getenv('DATA_DIR', os.path.join(BASE_DIR, 'data'))
TEMPLATES_DIR = os.getenv('TEMPLATES_DIR', os.path.join(BASE_DIR, 'templates'))

print(f"Model directory: {MODEL_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Templates directory: {TEMPLATES_DIR}")

app = FastAPI()
app.include_router(router1)
app.include_router(router2)      # 모델 라우터

templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

