from fastapi import Request
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from routers.predict_router import router2, router1
import mysql.connector

app = FastAPI()
app.include_router(router1)
app.include_router(router2)      # 모델 라우터

templates = Jinja2Templates(directory="/Users/jeonghaechan/projects/capstone-fastapi/templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

