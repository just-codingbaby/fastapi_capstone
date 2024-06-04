import os
import sys

import httpx
from pydantic import BaseModel

# 현재 파일의 디렉토리 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 디렉토리를 sys.path에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
env_path = os.path.join(PROJECT_ROOT, '.env')
if not os.path.exists(env_path):
    raise FileNotFoundError(f".env file not found at path: {env_path}")

load_dotenv(env_path)

MODEL_DIR = os.getenv('MODEL_DIR')
DATA_DIR = os.getenv('DATA_DIR')
TEMPLATES_DIR = os.getenv('TEMPLATES_DIR', os.path.join(PROJECT_ROOT, 'templates'))
print(f"PROJECT ROOT:{PROJECT_ROOT}")
print(f"Model directory: {MODEL_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Templates directory: {TEMPLATES_DIR}")

app = FastAPI()

# 라우터 경로 설정
from routers.predict_router import router2, router1

app.include_router(router1)
app.include_router(router2)  # 모델 라우터

templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/test")
async def receive_from_spring(data: str):
    return {"message": f"Received data: {data}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
