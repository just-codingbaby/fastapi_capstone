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

spring_data = '''
{
  "distanceList": [
    4.1906956376112365,
    1.4999880249816928,
    1.2999896755071716,
    0.7999935999898892,
    1.4999880249798299,
    2.399980999967977,
    0.999992049986346,
    0.799993699990206,
    0.6999944249899556,
    0.4999960749914655,
    0.0,
    2.988896762923905,
    0.9636437960366208,
    0.9999921267900431,
    0.9927639513360413,
    0.0,
    1.5606461654008557,
    1.9895641668890367,
    0.0,
    0.09986617431406479,
    1.6874436576274225,
    0.5993819975892135,
    1.0986389677619552,
    1.066936518164177,
    0.0,
    0.0,
    1.4538988571101652,
    1.1987468908573882,
    1.0938749232991716,
    1.8966950517453882,
    0.9989703079347584,
    0.899076920980936,
    0.8978297351193796,
    1.1806465013934393,
    0.7991746296422814,
    0.8990734765230873,
    1.3969069423869225,
    0.0,
    0.09989759843413354,
    0.0,
    0.0,
    0.0,
    2.5919373425013412,
    0.0,
    0.49948394595285434,
    0.5986928252728463,
    0.7991707642965955,
    0.0,
    0.09989752521446697,
    0.399590210051687,
    1.293467890429736,
    2.797106544125241,
    1.798128949510761,
    1.465576530253371,
    1.4984577338157528,
    0.9989648836298549,
    3.990972544729961,
    0.8990711802909805,
    0.7991101029410693,
    2.1935245682543445,
    1.866014934885366,
    0.0,
    0.9848881297428189,
    1.3985654114589898,
    0.7930649090455039,
    1.893700712246676,
    1.1976069018795732,
    0.6859419855381079,
    0.6892128955427766,
    0.8990644367632292,
    1.7555863563983098,
    0.0,
    0.0,
    1.988120806599329,
    2.1529743884829364,
    1.2772937815162897,
    1.6973647265698877,
    1.2593936520778877,
    1.0988121532205364,
    6.809932987122162,
    1.4578059033454132,
    2.5010283432127403,
    2.2974122920397653,
    0.5992497744126568,
    0.0,
    8.07898042170761,
    0.0,
    1.2880578194034413,
    1.1951731359608684,
    1.1036572963753688,
    2.1099786725872782,
    4.671062170840599,
    3.7778514452655125,
    1.0028982552066266,
    3.6292260362131143,
    5.781796443582117
  ]
}
'''

data = json.loads(spring_data)
l = data["distanceList"]




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
        start = time.time()
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

            distance_node = l[i]
            t = distance_node / pred[0][-1] * 60        #분으로 변경
            list_t.append(t)

            print("distance: %f, avg_speed: %f , time: %f" % (distance_node,pred[0][-1], t))

            if i == 95:
                print(sum(list_t))
                continue
            i+=1

        end = time.time()
        print(end-start)
        return {"result": result}
    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e))



