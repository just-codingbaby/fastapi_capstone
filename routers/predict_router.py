from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model.inference import SegRnn, use_cuda,create_dataframe_from_table
import pandas as pd
import torch
import mysql.connector
import os
import json
import time

from app.db_location import connect_to_mysql

router = APIRouter()

connection = connect_to_mysql()
print("connect success")

device = torch.device('cuda' if use_cuda else 'cpu')

class PredictionInput(BaseModel):       # 입력 데이터(YYYY-MM-DD-HH-MM-SS)
    data: str

# 파일이름:노드명_위도_경도
directory = "/Users/jeonghaechan/projects/capstone-fastapi/data/경부선_gps"
files = sorted([file for file in os.listdir(directory) if not file.startswith('.DS_Store')])

@router.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)
        print(end_time)

        timedelta = pd.Timedelta(hours=4)
        start_time = end_time - timedelta
        print(start_time)


        result = []     #예측결과 리스트로 받음
        i = 0           #전달받은 경로 사이의 거리 리스트 요소 갯수만큼 for문에서 ++
        list_t = []     #시간 리스트
        for file in files:
            if file[:12] < '0010VDS01600' or file[:12] > '0010VDS11200':
                continue

            print(file[:12])
            df = create_dataframe_from_table(file[:12])     # 데이터프레임 형성

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]

            model_test = pd.DataFrame(filtered_df).to_numpy()
            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)

            model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/경부선모델/' + file[:12] + '.pkl'

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)


            pred = prediction[:, -1, :].cpu().numpy()
            # print(pred[0][-1])
            result.append(float(pred[0][-1]))


            # t = distance_node / pred[0][-1] * 60        #분으로 변경
            # list_t.append(t)

            if i == 95:
                print(sum(list_t))
                continue
            i+=1

        return {"result": result}
    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e))



