import asyncio
import logging
import os
import sys
from fastapi import HTTPException

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
CSV_DIR = os.getenv('CSV_DIR')
print(f"PROJECT ROOT:{PROJECT_ROOT}")
print(f"Model directory: {MODEL_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Csv directory: {CSV_DIR}")
print(f"Templates directory: {TEMPLATES_DIR}")


app = FastAPI()

# 라우터 경로 설정
from routers.predict_router import router1, router2

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PredictionInput(BaseModel):       # 입력 데이터(YYYY-MM-DD-HH-MM-SS)
    data: str
    start: str

@app.post("/predict")
async def predict(input_data: PredictionInput):
    logger.debug(f"Received input data: {input_data}")
    try:
        results = await asyncio.gather(
            router1(input_data),
            router2(input_data)
        )
        logger.debug(f"Response: {results}")

        return {
            "RouteA Time": results[0],
            "RouteB Time": results[1][0],
            "RouteC Time": results[1][1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
