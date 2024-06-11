import datetime

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model.inference import use_cuda, create_dataframe_from_table, SegRnn
import pandas as pd
import torch
import mysql.connector
import os
import json
import time

load_dotenv()

MODEL_DIR = os.getenv('MODEL_DIR')
DATA_DIR = os.getenv('DATA_DIR')
CSV_DIR = os.getenv('CSV_DIR')

from app.db_process import connect_to_mysql

router1 = APIRouter()
router2 = APIRouter()
router3 = APIRouter()
router4 = APIRouter()
router5 = APIRouter()
router6 = APIRouter()


device = torch.device('cuda' if use_cuda else 'cpu')

class PredictionInput(BaseModel):       # 입력 데이터(YYYY-MM-DD-HH-MM-SS)
    data: str

# 파일이름:노드명_위도_경도

# 중부-경부 경로1: RouteA
@router1.post("/predict_router1")
async def predict(input_data: PredictionInput):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)

        # csv에 들어갈 시간
        csv_time = end_time - pd.DateOffset(months=1) + pd.DateOffset(days=10)
        print(f"2번 csv_time: {csv_time}")
        csv_file_path = os.path.join(CSV_DIR, '2번실시간.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time]
        duration = int(matched_row['duration'].values[0])
        print(f"2번 duration: {duration}")

        #모델에 들어갈 시간
        end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)

        print(end_time)

        timedelta = pd.Timedelta(hours=4)
        start_time = end_time - timedelta
        print(start_time)

        result_r1 = []  # 예측결과 리스트로 받음
        i = 0  # 전달받은 경로 사이의 거리 리스트 요소 갯수만큼 for문에서 ++
        list_t = []  # 시간 리스트

        # 중부-경부 중에서 중부 먼저

        connection_r1 = connect_to_mysql("경로")
        cursor_r1 = connection_r1.cursor(dictionary=True)
        query = "SELECT 노드명,위도, 경도, 거리 FROM 경로1 where 노드명 Like %s"  # 테이블에 자꾸 경부선이 먼저 올라가서 이렇게
        cursor_r1.execute(query, ('0352VDS%',))
        records_r1_0352 = cursor_r1.fetchall()

        print("경로1 중부선 실행 시작")
        start = time.time()
        for node in records_r1_0352:
            node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
            print(node_name)

            df = create_dataframe_from_table(node_name, "중부선")  # 데이터프레임 형성
            # -4로 하는 건 .Csv 제거

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
            model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()
            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
            # model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/중부선모델/' + node["노드명"] + '.pkl'
            # 모델 디렉토리 나중에 변동해야함
            model_name = os.path.join(MODEL_DIR, '중부선모델', node["노드명"] + '.pkl')

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)

            pred = prediction[:, -1, :].cpu().numpy()
            # print(pred[0][-1])
            result_r1.append(float(pred[0][-1]))

            distance = node["거리"]
            if distance is None:
                continue
            t = distance / 1000 / pred[0][-1] * 60  # km로 바꾸고, 분으로 변경
            # print(f"거리:{distance} | 속도:{pred[0][-1]}|시간:{t}")
            list_t.append(t)
        print("경로1 중부선 실행 완료")

        cursor_r1.execute(query, ('0010VDS%',))
        records_r1_0010 = cursor_r1.fetchall()

        print("경로1 경부선 실행 시작")
        for node in records_r1_0010:
            node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
            print(node_name)

            df = create_dataframe_from_table(node_name, "경부선")  # 데이터프레임 형성
            # -4로 하는 건 .Csv 제거

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
            model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()

            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)

            # model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/경부선모델/' + node["노드명"] + '.pkl'
            model_name = os.path.join(MODEL_DIR, '경부선모델', node["노드명"] + '.pkl')
            # 모델 디렉토리 나중에 변동해야함

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)

            pred = prediction[:, -1, :].cpu().numpy()
            # print(pred[0][-1])
            result_r1.append(float(pred[0][-1]))

            distance = node["거리"]
            if distance is None:
                continue
            t = distance / 1000 / pred[0][-1] * 60  # km로 바꾸고, 분으로 변경
            # print(f"거리:{distance} | 속도:{pred[0][-1]} | 시간:{t}")

            list_t.append(t)

        print("경로1 경부선 실행 완료")
        end = time.time()
        print(f"모델 실행 시간: {end-start}")
        time_route1 = int(round(sum(list_t),2))
        print(f"예측 결과 시간: {time_route1}분")

        csv_time_route1 = csv_time + datetime.timedelta(minutes=time_route1)
        print(f"6번 csv_time: {csv_time_route1}")
        csv_file_path = os.path.join(CSV_DIR, '6번실시간.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'],encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time_route1]
        duration2 = int(matched_row['duration'].values[0])
        print(f"6번 duration: {duration2}")

        time_route1 = time_route1 + duration + duration2

        return {"RouteA Time": time_route1}

    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e))

#경부선 경로3, 실시간추천 RouterC
@router2.post("/predict_router3")
async def predict(input_data: PredictionInput):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)

        # csv에 들어갈 시간
        csv_time = end_time - pd.DateOffset(months=1) - pd.DateOffset(days=4)
        print(f"4번 csv_time: {csv_time}")
        csv_file_path = os.path.join(CSV_DIR, '4번실시간.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time]
        duration = int(matched_row['duration'].values[0])
        print(f"4번 duration: {duration}")

        end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)

        print(end_time)

        timedelta = pd.Timedelta(hours=4)
        start_time = end_time - timedelta
        print(start_time)

        result = []     #예측결과 리스트로 받음
        i = 0           #전달받은 경로 사이의 거리 리스트 요소 갯수만큼 for문에서 ++
        list_t = []     #시간 리스트

        connection_r0010 = connect_to_mysql("경로")
        cursor_r0010 = connection_r0010.cursor(dictionary=True)
        query = "SELECT 노드명,위도, 경도, 거리 FROM 경로2"
        cursor_r0010.execute(query)
        records_r0010 = cursor_r0010.fetchall()

        print("경로2 실행 시작")
        start = time.time()
        for node in records_r0010:
            node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
            print(node_name)

            df = create_dataframe_from_table(node_name,"경부선")     # 데이터프레임 형성

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
            model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()

            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)

            # model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/경부선모델/' + node["노드명"] + '.pkl'
            model_name = os.path.join(MODEL_DIR, '경부선모델', node["노드명"] + '.pkl')
            # 모델 디렉토리 나중에 변동해야함

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)


            pred = prediction[:, -1, :].cpu().numpy()
            result.append(float(pred[0][-1]))

            distance = node["거리"]
            if distance is None:
                continue
            t = distance / 1000 / pred[0][-1] * 60  # km로 바꾸고, 분으로 변경
            list_t.append(t)

        end=time.time()
        print("경로2 종료")
        print(f"모델 실행 시간: {end-start}")
        time_route2 = int(round(sum(list_t),2))
        print(f"예측 결과 시간: {time_route2}분")

        csv_time_route2 = csv_time + datetime.timedelta(minutes=time_route2) + datetime.timedelta(days=14)
        print(f"6번 csv_time: {csv_time_route2}")
        csv_file_path = os.path.join(CSV_DIR, '6번실시간.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time_route2]
        duration2 = int(matched_row['duration'].values[0])
        print(f"6번 duration: {duration2}")

        time_route2 = time_route2 + duration + duration2


        return {"RouteC time": time_route2}
    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e)) 

#경부선 경로2 최단경로 RouterB
@router3.post("/predict_router2")
async def predict(input_data: PredictionInput):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)

        # csv에 들어갈 시간
        csv_time = end_time - pd.DateOffset(months=1) + pd.DateOffset(days=10)
        print(f"3번 csv_time: {csv_time}")
        csv_file_path = os.path.join(CSV_DIR, '3번최단거리.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time]
        duration = int(matched_row['duration'].values[0])
        print(f"4번 duration: {duration}")

        end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)

        print(end_time)

        timedelta = pd.Timedelta(hours=4)
        start_time = end_time - timedelta
        print(start_time)

        result = []     #예측결과 리스트로 받음
        i = 0           #전달받은 경로 사이의 거리 리스트 요소 갯수만큼 for문에서 ++
        list_t = []     #시간 리스트

        connection_r0010 = connect_to_mysql("경로")
        cursor_r0010 = connection_r0010.cursor(dictionary=True)
        query = "SELECT 노드명,위도, 경도, 거리 FROM 경로2"
        cursor_r0010.execute(query)
        records_r0010 = cursor_r0010.fetchall()

        print("경로2 실행 시작")
        start = time.time()
        for node in records_r0010:
            node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
            print(node_name)

            df = create_dataframe_from_table(node_name,"경부선")     # 데이터프레임 형성

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
            model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()

            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)

            # model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/경부선모델/' + node["노드명"] + '.pkl'
            model_name = os.path.join(MODEL_DIR, '경부선모델', node["노드명"] + '.pkl')
            # 모델 디렉토리 나중에 변동해야함

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)


            pred = prediction[:, -1, :].cpu().numpy()
            result.append(float(pred[0][-1]))

            distance = node["거리"]
            if distance is None:
                continue
            t = distance / 1000 / pred[0][-1] * 60  # km로 바꾸고, 분으로 변경
            list_t.append(t)

        end=time.time()
        print("경로2 종료")
        print(f"모델 실행 시간: {end-start}")
        time_route2 = int(round(sum(list_t),2))
        print(f"예측 결과 시간: {time_route2}분")

        csv_time_route2 = csv_time + datetime.timedelta(minutes=time_route2)
        print(f"5번 csv_time: {csv_time_route2}")
        csv_file_path = os.path.join(CSV_DIR, '5번최단거리.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time_route2]
        duration2 = int(matched_row['duration'].values[0])
        print(f"6번 duration: {duration2}")

        time_route2 = time_route2 + duration + duration2


        return {"RouteB time": time_route2}
    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e))


class ReturnInput(BaseModel):
    data:str

#중부+경부 성심당->세종대
@router4.post("/reverse_router1")
async def predict(input_data: ReturnInput):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)

        #csv에 들어갈 시간
        csv_time = end_time - pd.DateOffset(months=1) + pd.DateOffset(days=10)
        print(f"6번 csv_time: {csv_time}")
        csv_file_path = os.path.join(CSV_DIR, '6번실시간.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time]
        duration = int(matched_row['duration'].values[0])
        print(f"2번 duration: {duration}")

        #모델에 들어갈 시간
        end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)

        print(end_time)

        timedelta = pd.Timedelta(hours=4)
        start_time = end_time - timedelta
        print(start_time)

        result_r1 = []  # 예측결과 리스트로 받음
        i = 0  # 전달받은 경로 사이의 거리 리스트 요소 갯수만큼 for문에서 ++
        list_t = []  # 시간 리스트

        # 중부-경부 중에서 중부 먼저

        connection_r1 = connect_to_mysql("경로")
        cursor_r1 = connection_r1.cursor(dictionary=True)
        query = "SELECT 노드명,위도, 경도, 거리 FROM 역방향1 where 노드명 Like %s"  # 테이블에 자꾸 경부선이 먼저 올라가서 이렇게
        cursor_r1.execute(query, ('0010VDE%',))
        records_r1_0352 = cursor_r1.fetchall()

        print("경로1 경부선 실행 시작 / 성심당->세종대")
        start = time.time()
        for node in records_r1_0352:
            node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
            print(node_name)

            df = create_dataframe_from_table(node_name, "경부선_VDE")  # 데이터프레임 형성
            # -4로 하는 건 .Csv 제거

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
            model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()
            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
            # model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/중부선모델/' + node["노드명"] + '.pkl'
            # 모델 디렉토리 나중에 변동해야함
            model_name = os.path.join(MODEL_DIR, '경부선VDE모델', node["노드명"] + '.pkl')

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)

            pred = prediction[:, -1, :].cpu().numpy()
            # print(pred[0][-1])
            result_r1.append(float(pred[0][-1]))

            distance = node["거리"]
            if distance is None:
                continue
            t = distance / 1000 / pred[0][-1] * 60  # km로 바꾸고, 분으로 변경
            # print(f"거리:{distance} | 속도:{pred[0][-1]}|시간:{t}")
            list_t.append(t)
        print("경로1 경부선 실행 완료 / 성심당->세종대")

        cursor_r1.execute(query, ('0352VDE%',))
        records_r1_0010 = cursor_r1.fetchall()

        print("경로1 중부선 실행 시작 / 성심당->세종대")
        for node in records_r1_0010:
            node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
            print(node_name)

            df = create_dataframe_from_table(node_name, "중부선_VDE")  # 데이터프레임 형성
            # -4로 하는 건 .Csv 제거

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
            model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()

            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)

            # model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/경부선모델/' + node["노드명"] + '.pkl'
            model_name = os.path.join(MODEL_DIR, '중부선VDE모델', node["노드명"] + '.pkl')
            # 모델 디렉토리 나중에 변동해야함

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)

            pred = prediction[:, -1, :].cpu().numpy()
            # print(pred[0][-1])
            result_r1.append(float(pred[0][-1]))

            distance = node["거리"]
            if distance is None:
                continue
            t = distance / 1000 / pred[0][-1] * 60  # km로 바꾸고, 분으로 변경
            # print(f"거리:{distance} | 속도:{pred[0][-1]} | 시간:{t}")

            list_t.append(t)

        print("경로1 중부선 실행 완료 / 성심당->세종대")
        end = time.time()
        print(f"모델 실행 시간: {end-start}")
        time_reverse1 = int(round(sum(list_t),2))
        print(f"예측 결과 시간: {time_reverse1}분")

        csv_time_route1 = csv_time + datetime.timedelta(minutes=time_reverse1)
        print(f"2번 csv_time: {csv_time_route1}")
        csv_file_path = os.path.join(CSV_DIR, '2번실시간.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'],encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time_route1]
        duration2 = int(matched_row['duration'].values[0])
        print(f"2번 duration: {duration2}")

        time_reverse1 = time_reverse1 + duration + duration2

        return {"ReverseA Time": time_reverse1}

    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e))

#경부선 실시간추천 경로3 / 성심당->세종대
@router5.post("/reverse_router3")
async def predict(input_data: ReturnInput):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)

        #csv에 들어갈 시간
        csv_time = end_time - pd.DateOffset(months=1) + pd.DateOffset(days=10)
        print(f"6번 csv_time: {csv_time}")
        csv_file_path = os.path.join(CSV_DIR, '6번실시간.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time]
        duration = int(matched_row['duration'].values[0])
        print(f"6번 duration: {duration}")

        end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)

        print(end_time)

        timedelta = pd.Timedelta(hours=4)
        start_time = end_time - timedelta
        print(start_time)

        result = []     #예측결과 리스트로 받음
        i = 0           #전달받은 경로 사이의 거리 리스트 요소 갯수만큼 for문에서 ++
        list_t = []     #시간 리스트

        connection_r0010 = connect_to_mysql("경로")
        cursor_r0010 = connection_r0010.cursor(dictionary=True)
        query = "SELECT 노드명,위도, 경도, 거리 FROM 역방향2"
        cursor_r0010.execute(query)
        records_r0010 = cursor_r0010.fetchall()

        print("경로2 실행 시작 성심당->세종대")
        start = time.time()
        for node in records_r0010:
            node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
            print(node_name)

            df = create_dataframe_from_table(node_name,"경부선_VDE")     # 데이터프레임 형성

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
            model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()

            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)

            # model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/경부선모델/' + node["노드명"] + '.pkl'
            model_name = os.path.join(MODEL_DIR, '경부선VDE모델', node["노드명"] + '.pkl')
            # 모델 디렉토리 나중에 변동해야함

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)


            pred = prediction[:, -1, :].cpu().numpy()
            result.append(float(pred[0][-1]))

            distance = node["거리"]
            if distance is None:
                continue
            t = distance / 1000 / pred[0][-1] * 60  # km로 바꾸고, 분으로 변경
            list_t.append(t)

        end=time.time()
        print("경로2 종료 성심당->세종대")
        print(f"모델 실행 시간: {end-start}")
        time_reverse2 = int(round(sum(list_t),2))
        print(f"예측 결과 시간: {time_reverse2}분")

        csv_time_route2 = csv_time + datetime.timedelta(minutes=time_reverse2) - datetime.timedelta(days=14)
        print(f"4번 csv_time: {csv_time_route2}")
        csv_file_path = os.path.join(CSV_DIR, '4번실시간.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time_route2]
        duration2 = int(matched_row['duration'].values[0])
        print(f"4번 duration: {duration2}")

        time_reverse2 = time_reverse2 + duration + duration2


        return {"ReverseC time": time_reverse2}
    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e))

#경로2 경부선 최단경로 성심당->세종대
@router6.post("/reverse_router2")
async def predict(input_data: PredictionInput):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)

        # csv에 들어갈 시간
        csv_time = end_time - pd.DateOffset(months=1) + pd.DateOffset(days=10)
        print(f"5번 csv_time: {csv_time}")
        csv_file_path = os.path.join(CSV_DIR, '5번최단거리.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time]
        duration = int(matched_row['duration'].values[0])
        print(f"5번 duration: {duration}")

        end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)

        print(end_time)

        timedelta = pd.Timedelta(hours=4)
        start_time = end_time - timedelta
        print(start_time)

        result = []     #예측결과 리스트로 받음
        i = 0           #전달받은 경로 사이의 거리 리스트 요소 갯수만큼 for문에서 ++
        list_t = []     #시간 리스트

        connection_r0010 = connect_to_mysql("경로")
        cursor_r0010 = connection_r0010.cursor(dictionary=True)
        query = "SELECT 노드명,위도, 경도, 거리 FROM 역방향2"
        cursor_r0010.execute(query)
        records_r0010 = cursor_r0010.fetchall()

        print("경로2 실행 시작 / 성심당->세종대")
        start = time.time()
        for node in records_r0010:
            node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
            print(node_name)

            df = create_dataframe_from_table(node_name,"경부선_VDE")     # 데이터프레임 형성

            enc_in = len(df.columns)
            seq_len = 48
            pred_len = 1
            patch_len = 1
            dropout = 0.1
            hidden_dim = 32

            # input 값에 따라 필터링
            filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
            model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()

            model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)

            # model_name = '/Users/jeonghaechan/projects/capstone-fastapi/model/경부선모델/' + node["노드명"] + '.pkl'
            model_name = os.path.join(MODEL_DIR, '경부선VDE모델', node["노드명"] + '.pkl')
            # 모델 디렉토리 나중에 변동해야함

            example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
            example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

            example.eval()

            # 모델 추론 수행
            with torch.no_grad():
                prediction = example(model_test_scaled)


            pred = prediction[:, -1, :].cpu().numpy()
            result.append(float(pred[0][-1]))

            distance = node["거리"]
            if distance is None:
                continue
            t = distance / 1000 / pred[0][-1] * 60  # km로 바꾸고, 분으로 변경
            list_t.append(t)

        end=time.time()
        print("경로2 종료 / 성심당->세종대")
        print(f"모델 실행 시간: {end-start}")
        time_route2 = int(round(sum(list_t),2))
        print(f"예측 결과 시간: {time_route2}분")

        csv_time_route2 = csv_time + datetime.timedelta(minutes=time_route2)
        print(f"3번 csv_time: {csv_time_route2}")
        csv_file_path = os.path.join(CSV_DIR, '3번최단거리.csv')
        df = pd.read_csv(csv_file_path, parse_dates=['time'], encoding='EUC-KR')
        matched_row = df[df['time'] == csv_time_route2]
        duration2 = int(matched_row['duration'].values[0])
        print(f"3번 duration: {duration2}")

        time_route2 = time_route2 + duration + duration2


        return {"ReverseB time": time_route2}
    except Exception as e:
        # 예외 발생 시 HTTP 예외 반환
        raise HTTPException(status_code=500, detail=str(e))
