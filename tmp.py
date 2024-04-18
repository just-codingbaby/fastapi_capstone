from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model.inference import SegRnn, use_cuda
import pandas as pd
import torch
import mysql.connector
import os

router = APIRouter()

# directory = "/Users/jeonghaechan/Desktop/capstone/0010경부선"
# files = sorted([file for file in os.listdir(directory) if not file.startswith('.DS_Store')])

# connection = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='Wjdgocks2!',
#     database='경부선'
# )
#
# def create_dataframe_from_table(table_name):
#     cursor = connection.cursor()
#     cursor.execute(f"SELECT * FROM {table_name}")
#     rows = cursor.fetchall()
#     df = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
#     cursor.close()
#     return df

path = '/Users/jeonghaechan/projects/capstone-fastapi/data/' #여기 수정해야 함
df = pd.read_csv(path + '0251VDS18000.csv') #csv 파일은 노드마다 바뀌기에 이 부분도 수정해야 함
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].apply(lambda row:row.year, 1)
df['month'] = df['datetime'].apply(lambda row:row.month, 1)
df['date'] = df['datetime'].apply(lambda row:row.day, 1)
df['hour'] = df['datetime'].apply(lambda row:row.hour, 1)
df['minute'] = df['datetime'].apply(lambda row:row.minute, 1)
df['weekday'] = df['datetime'].apply(lambda row:row.weekday(), 1)

df.set_index('datetime', drop = True, inplace = True)
df = df[['TRFFCVLM', 'OCCPNCY', 'year', 'month', 'date', 'hour', 'minute', 'weekday', 'SPD_AVG']]

enc_in = len(df.columns)
# seq_len = 92
pred_len = 1
patch_len = 1
dropout = 0.1
hidden_dim = 128

device = torch.device('cuda' if use_cuda else 'cpu')

class PredictionInput(BaseModel):
    data: str

@router.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        input_string = input_data.data
        input_string = pd.to_datetime(input_string)

        # 입력 들어왔는지 확인
        print(input_string)

        filtered_df = df[(df['hour'] == input_string.hour) & (df['minute'] == input_string.minute)]

        model_test = pd.DataFrame(filtered_df).to_numpy()
        model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)

        # 입력 데이터 길이에 따라 seq_len 설정
        seq_len = len(filtered_df)

        # 모델 생성 및 가중치 로드
        example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
        example.load_state_dict(torch.load('/Users/jeonghaechan/projects/capstone-fastapi/model/example.pkl',
                                           map_location=torch.device('cpu')))

        # 추론 모드 설정
        example.eval()

        # 모델 추론 수행
        with torch.no_grad():
            prediction = example(model_test_scaled)
            # print("prediction"+ str(prediction))

        pred = prediction[:, -1, :].cpu().numpy()
        result = float(pred[0][-1])

        return {"result": result}
    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e))
