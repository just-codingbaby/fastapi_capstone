import asyncio
import datetime
from dotenv import load_dotenv
from fastapi import HTTPException
import pandas as pd
import torch
import os
import time
from model.inference import use_cuda, create_dataframe_from_table, SegRnn
from app.db_process import connect_to_mysql

load_dotenv()

MODEL_DIR = os.getenv('MODEL_DIR')
DATA_DIR = os.getenv('DATA_DIR')
CSV_DIR = os.getenv('CSV_DIR')

device = torch.device('cuda' if use_cuda else 'cpu')


async def router1(input_data):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)
        start = input_data.start

        if start == 'A':
            csv_time = end_time - pd.DateOffset(months=1) + pd.DateOffset(days=10)
            print(f"2번 csv_time: {csv_time}")
            csv_file_path = os.path.join(CSV_DIR, '2번실시간.csv')
            df = await asyncio.to_thread(pd.read_csv, csv_file_path, parse_dates=['time'], encoding='EUC-KR')
            matched_row = df[df['time'] == csv_time]
            duration = int(matched_row['duration'].values[0])
            print(f"2번 duration: {duration}")

            end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)
            print(end_time)

            timedelta = pd.Timedelta(hours=4)
            start_time = end_time - timedelta
            print(start_time)

            result_r1 = []
            list_t = []

            connection_r1 = await asyncio.to_thread(connect_to_mysql, "경로")
            cursor_r1 = connection_r1.cursor(dictionary=True)
            query = "SELECT 노드명,위도, 경도, 거리 FROM 경로1 where 노드명 Like %s"
            await asyncio.to_thread(cursor_r1.execute, query, ('0352VDS%',))
            records_r1_0352 = await asyncio.to_thread(cursor_r1.fetchall)

            print("경로1 중부선 실행 시작")
            start_time_exec = time.time()
            for node in records_r1_0352:
                node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
                print(node_name)

                df = await asyncio.to_thread(create_dataframe_from_table, node_name, "중부선")
                enc_in = len(df.columns)
                seq_len = 48
                pred_len = 1
                patch_len = 1
                dropout = 0.1
                hidden_dim = 32

                df.index = pd.to_datetime(df.index)
                print(f"df.index dtype: {df.index.dtype}")
                print(f"start_time: {start_time}, end_time: {end_time}")

                filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
                print(f"Filtered dataframe size: {filtered_df.shape}")
                model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()
                model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
                model_name = os.path.join(MODEL_DIR, '중부선모델', node["노드명"] + '.pkl')

                example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
                example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

                example.eval()

                with torch.no_grad():
                    prediction = example(model_test_scaled)

                pred = prediction[:, -1, :].cpu().numpy()
                result_r1.append(float(pred[0][-1]))

                distance = node["거리"]
                if distance is None:
                    continue
                t = distance / 1000 / pred[0][-1] * 60
                list_t.append(t)
            print("경로1 중부선 실행 완료")

            cursor_r1.execute(query, ('0010VDS%',))
            records_r1_0010 = cursor_r1.fetchall()

            print("경로1 경부선 실행 시작")
            for node in records_r1_0010:
                node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
                print(node_name)

                df = await asyncio.to_thread(create_dataframe_from_table, node_name, "경부선")
                enc_in = len(df.columns)
                seq_len = 48
                pred_len = 1
                patch_len = 1
                dropout = 0.1
                hidden_dim = 32

                df.index = pd.to_datetime(df.index)
                print(f"df.index dtype: {df.index.dtype}")
                print(f"start_time: {start_time}, end_time: {end_time}")

                filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
                print(f"Filtered dataframe size: {filtered_df.shape}")
                model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()
                model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
                model_name = os.path.join(MODEL_DIR, '경부선모델', node["노드명"] + '.pkl')

                example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
                example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

                example.eval()

                with torch.no_grad():
                    prediction = example(model_test_scaled)

                pred = prediction[:, -1, :].cpu().numpy()
                result_r1.append(float(pred[0][-1]))

                distance = node["거리"]
                if distance is None:
                    continue
                t = distance / 1000 / pred[0][-1] * 60
                list_t.append(t)

            print("경로1 경부선 실행 완료")
            end_time_exec = time.time()
            print(f"모델 실행 시간: {end_time_exec - start_time_exec}")
            time_route1 = int(round(sum(list_t), 2))
            print(f"경로1 고속도로 예측 결과 시간: {time_route1}분")

            csv_time_route1 = csv_time + datetime.timedelta(minutes=time_route1)
            print(f"6번 csv_time: {csv_time_route1}")
            csv_file_path = os.path.join(CSV_DIR, '6번실시간.csv')
            df = await asyncio.to_thread(pd.read_csv, csv_file_path, parse_dates=['time'], encoding='EUC-KR')
            matched_row = df[df['time'] == csv_time_route1]
            duration2 = int(matched_row['duration'].values[0])
            print(f"6번 duration: {duration2}")

            time_route1 = time_route1 + duration + duration2

        else:
            csv_time = end_time - pd.DateOffset(days=7)
            print(f"6번 csv_time: {csv_time}")
            csv_file_path = os.path.join(CSV_DIR, '6번실시간VDE.csv')
            df = await asyncio.to_thread(pd.read_csv, csv_file_path, parse_dates=['time'], encoding='EUC-KR')
            matched_row = df[df['time'] == csv_time]
            duration = int(matched_row['duration'].values[0])
            print(f"6번 duration: {duration}")

            end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)
            print(end_time)

            timedelta = pd.Timedelta(hours=4)
            start_time = end_time - timedelta
            print(start_time)

            result_r1 = []
            list_t = []

            connection_r1 = await asyncio.to_thread(connect_to_mysql, "경로")
            cursor_r1 = connection_r1.cursor(dictionary=True)
            query = "SELECT 노드명,위도, 경도, 거리 FROM 역방향1 where 노드명 Like %s"
            await asyncio.to_thread(cursor_r1.execute, query, ('0010VDE%',))
            records_r1_0352 = await asyncio.to_thread(cursor_r1.fetchall)

            print("경로1 경부선 실행 시작 / 성심당->세종대")
            start_time_exec = time.time()
            for node in records_r1_0352:
                node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
                print(node_name)

                df = await asyncio.to_thread(create_dataframe_from_table, node_name, "경부선_VDE")
                enc_in = len(df.columns)
                seq_len = 48
                pred_len = 1
                patch_len = 1
                dropout = 0.1
                hidden_dim = 32

                df.index = pd.to_datetime(df.index)
                print(f"df.index dtype: {df.index.dtype}")
                print(f"start_time: {start_time}, end_time: {end_time}")

                filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
                print(f"Filtered dataframe size: {filtered_df.shape}")
                model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()
                model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
                model_name = os.path.join(MODEL_DIR, '경부선VDE모델', node["노드명"] + '.pkl')

                example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
                example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

                example.eval()

                with torch.no_grad():
                    prediction = example(model_test_scaled)

                pred = prediction[:, -1, :].cpu().numpy()
                result_r1.append(float(pred[0][-1]))

                distance = node["거리"]
                if distance is None:
                    continue
                t = distance / 1000 / pred[0][-1] * 60
                list_t.append(t)
            print("경로1 경부선 실행 완료 / 성심당->세종대")

            cursor_r1.execute(query, ('0352VDE%',))
            records_r1_0010 = cursor_r1.fetchall()

            print("경로1 중부선 실행 시작 / 성심당->세종대")
            for node in records_r1_0010:
                node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
                print(node_name)

                df = await asyncio.to_thread(create_dataframe_from_table, node_name, "중부선_VDE")
                enc_in = len(df.columns)
                seq_len = 48
                pred_len = 1
                patch_len = 1
                dropout = 0.1
                hidden_dim = 32

                df.index = pd.to_datetime(df.index)
                print(f"df.index dtype: {df.index.dtype}")
                print(f"start_time: {start_time}, end_time: {end_time}")

                filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
                print(f"Filtered dataframe size: {filtered_df.shape}")
                model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()
                model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
                model_name = os.path.join(MODEL_DIR, '중부선VDE모델', node["노드명"] + '.pkl')

                example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
                example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

                example.eval()

                with torch.no_grad():
                    prediction = example(model_test_scaled)

                pred = prediction[:, -1, :].cpu().numpy()
                result_r1.append(float(pred[0][-1]))

                distance = node["거리"]
                if distance is None:
                    continue
                t = distance / 1000 / pred[0][-1] * 60
                list_t.append(t)

            print("경로1 중부선 실행 완료 / 성심당->세종대")
            end_time_exec = time.time()
            print(f"모델 실행 시간: {end_time_exec - start_time_exec}")
            time_route1 = int(round(sum(list_t), 2))
            print(f"경로1 고속도로 예측 결과 시간: {time_route1}분")

            csv_time_route1 = csv_time + datetime.timedelta(minutes=time_route1)
            print(f"2번 csv_time: {csv_time_route1}")
            csv_file_path = os.path.join(CSV_DIR, '2번실시간VDE.csv')
            df = await asyncio.to_thread(pd.read_csv, csv_file_path, parse_dates=['time'], encoding='EUC-KR')
            matched_row = df[df['time'] == csv_time_route1]
            duration2 = int(matched_row['duration'].values[0])
            print(f"2번 duration: {duration2}")

            time_route1 = time_route1 + duration + duration2

        return time_route1

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


async def router2(input_data):
    try:
        end_time = input_data.data
        end_time = pd.to_datetime(end_time)
        start = input_data.start

        if start == 'A':
            csv_time = end_time - pd.DateOffset(days=7)
            print(f"경로2 3번 csv_time: {csv_time}")
            csv_file_path = os.path.join(CSV_DIR, '3번최단거리.csv')
            df = await asyncio.to_thread(pd.read_csv, csv_file_path, parse_dates=['time'], encoding='EUC-KR')
            matched_row = df[df['time'] == csv_time]
            duration = int(matched_row['duration'].values[0])
            print(f"경로2 3번 duration: {duration}")

            csv_time_3A = end_time - pd.DateOffset(months=1) - pd.DateOffset(days=4)
            print(f"경로3 4번 csv_time: {csv_time_3A}")
            csv_file_path_3A = os.path.join(CSV_DIR, '4번실시간.csv')
            df_3A = await asyncio.to_thread(pd.read_csv, csv_file_path_3A, parse_dates=['time'], encoding='EUC-KR')
            matched_row_3A = df_3A[df_3A['time'] == csv_time_3A]
            duration_3A = int(matched_row_3A['duration'].values[0])
            print(f"경로3 4번 duration: {duration_3A}")

            end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)
            print(end_time)

            timedelta = pd.Timedelta(hours=4)
            start_time = end_time - timedelta
            print(start_time)

            result = []
            list_t = []

            connection_r0010 = await asyncio.to_thread(connect_to_mysql, "경로")
            cursor_r0010 = connection_r0010.cursor(dictionary=True)
            query = "SELECT 노드명,위도, 경도, 거리 FROM 경로2"
            await asyncio.to_thread(cursor_r0010.execute, query)
            records_r0010 = await asyncio.to_thread(cursor_r0010.fetchall)

            print("라우터2 경부선 실행 시작")
            start_time_exec = time.time()
            for node in records_r0010:
                node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
                print(node_name)

                df = await asyncio.to_thread(create_dataframe_from_table, node_name, "경부선")
                enc_in = len(df.columns)
                seq_len = 48
                pred_len = 1
                patch_len = 1
                dropout = 0.1
                hidden_dim = 32

                df.index = pd.to_datetime(df.index)
                print(f"df.index dtype: {df.index.dtype}")
                print(f"start_time: {start_time}, end_time: {end_time}")

                filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
                print(f"Filtered dataframe size: {filtered_df.shape}")
                model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()
                model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
                model_name = os.path.join(MODEL_DIR, '경부선모델', node["노드명"] + '.pkl')

                example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
                example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

                example.eval()

                with torch.no_grad():
                    prediction = example(model_test_scaled)

                pred = prediction[:, -1, :].cpu().numpy()
                result.append(float(pred[0][-1]))

                distance = node["거리"]
                if distance is None:
                    continue
                t = distance / 1000 / pred[0][-1] * 60
                list_t.append(t)

            end_time_exec = time.time()
            print("라우터2 경부선 종료")
            print(f"모델 실행 시간: {end_time_exec - start_time_exec}")
            time_route2 = int(round(sum(list_t), 2))
            print(f"라우터2 고속도로 예측 결과 시간: {time_route2}분")

            csv_time_route2 = csv_time + datetime.timedelta(minutes=time_route2) - pd.DateOffset(
                months=1) + pd.DateOffset(days=14)
            print(f"경로2 5번 csv_time: {csv_time_route2}")
            csv_file_path = os.path.join(CSV_DIR, '5번최단거리.csv')
            df = await asyncio.to_thread(pd.read_csv, csv_file_path, parse_dates=['time'], encoding='EUC-KR')
            matched_row = df[df['time'] == csv_time_route2]
            duration2 = int(matched_row['duration'].values[0])
            print(f"경로2 5번 duration: {duration2}")

            csv_time_route3 = csv_time_3A + datetime.timedelta(minutes=time_route2) + datetime.timedelta(days=14)
            print(f"경로3 6번 csv_time: {csv_time_route3}")
            csv_file_path_3A = os.path.join(CSV_DIR, '6번실시간.csv')
            df_3A = await asyncio.to_thread(pd.read_csv, csv_file_path_3A, parse_dates=['time'], encoding='EUC-KR')
            matched_row_3A = df_3A[df_3A['time'] == csv_time_route3]
            duration2_3A = int(matched_row_3A['duration'].values[0])
            print(f"경로3 6번 duration: {duration2_3A}")

            time_route3 = time_route2 + duration_3A + duration2_3A
            time_route2 = time_route2 + duration + duration2

            result_route = []
            result_route.append(time_route2)
            result_route.append(time_route3)


        else:
            csv_time = end_time - pd.DateOffset(days=7)
            print(f"경로2 5번 csv_time: {csv_time}")
            csv_file_path = os.path.join(CSV_DIR, '5번최단거리VDE.csv')
            df = await asyncio.to_thread(pd.read_csv, csv_file_path, parse_dates=['time'], encoding='EUC-KR')
            matched_row = df[df['time'] == csv_time]
            duration = int(matched_row['duration'].values[0])
            print(f"경로2 5번 duration: {duration}")

            csv_time_3A = end_time - pd.DateOffset(days=7)
            print(f"경로3 6번 csv_time: {csv_time_3A}")
            csv_file_path_3A = os.path.join(CSV_DIR, '6번실시간VDE.csv')
            df_3A = await asyncio.to_thread(pd.read_csv, csv_file_path_3A, parse_dates=['time'], encoding='EUC-KR')
            matched_row_3A = df_3A[df_3A['time'] == csv_time_3A]
            duration_3A = int(matched_row_3A['duration'].values[0])
            print(f"경로3 6번 duration: {duration_3A}")

            end_time = end_time - pd.DateOffset(years=1) + pd.DateOffset(days=2)
            print(end_time)

            timedelta = pd.Timedelta(hours=4)
            start_time = end_time - timedelta
            print(start_time)

            result = []
            list_t = []

            connection_r0010 = await asyncio.to_thread(connect_to_mysql, "경로")
            cursor_r0010 = connection_r0010.cursor(dictionary=True)
            query = "SELECT 노드명,위도, 경도, 거리 FROM 역방향2"
            await asyncio.to_thread(cursor_r0010.execute, query)
            records_r0010 = await asyncio.to_thread(cursor_r0010.fetchall)

            print("경부선 실행 시작 / 성심당->세종대")
            start_time_exec = time.time()
            for node in records_r0010:
                node_name = f"{node['노드명']}_{str(node['위도'])}_{str(node['경도'])}"
                print(node_name)

                df = await asyncio.to_thread(create_dataframe_from_table, node_name, "경부선_VDE")
                enc_in = len(df.columns)
                seq_len = 48
                pred_len = 1
                patch_len = 1
                dropout = 0.1
                hidden_dim = 32

                df.index = pd.to_datetime(df.index)
                print(f"df.index dtype: {df.index.dtype}")
                print(f"start_time: {start_time}, end_time: {end_time}")

                filtered_df = df[(df.index >= start_time) & (df.index < end_time)]
                print(f"Filtered dataframe size: {filtered_df.shape}")
                model_test = pd.DataFrame(filtered_df).apply(pd.to_numeric, errors='coerce').to_numpy()
                model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
                model_name = os.path.join(MODEL_DIR, '경부선VDE모델', node["노드명"] + '.pkl')

                example = SegRnn(enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim).to(device)
                example.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

                example.eval()

                with torch.no_grad():
                    prediction = example(model_test_scaled)

                pred = prediction[:, -1, :].cpu().numpy()
                result.append(float(pred[0][-1]))

                distance = node["거리"]
                if distance is None:
                    continue
                t = distance / 1000 / pred[0][-1] * 60
                list_t.append(t)

            end_time_exec = time.time()
            print("경부선 종료 / 성심당->세종대")
            print(f"모델 실행 시간: {end_time_exec - start_time_exec}")
            time_route2 = int(round(sum(list_t), 2))
            print(f"경부선 고속도로 예측 결과 시간: {time_route2}분")

            csv_time_route2 = csv_time + datetime.timedelta(minutes=time_route2)
            print(f"경로2 3번 csv_time: {csv_time_route2}")
            csv_file_path = os.path.join(CSV_DIR, '3번최단거리VDE.csv')
            df = await asyncio.to_thread(pd.read_csv, csv_file_path, parse_dates=['time'], encoding='EUC-KR')
            matched_row = df[df['time'] == csv_time_route2]
            duration2 = int(matched_row['duration'].values[0])
            print(f"경로2 3번 duration: {duration2}")

            csv_time_route3 = csv_time_3A + datetime.timedelta(minutes=time_route2)
            print(f"경로3 4번 csv_time: {csv_time_route3}")
            csv_file_path_3A = os.path.join(CSV_DIR, '4번실시간VDE.csv')
            df_3A = await asyncio.to_thread(pd.read_csv, csv_file_path_3A, parse_dates=['time'], encoding='EUC-KR')
            matched_row_3A = df_3A[df_3A['time'] == csv_time_route3]
            duration2_3A = int(matched_row_3A['duration'].values[0])
            print(f"경로3 4번 duration: {duration2_3A}")

            time_route3 = time_route2 + duration_3A + duration2_3A
            time_route2 = time_route2 + duration + duration2

            result_route = []
            result_route.append(time_route2)        #경로2 최단거리
            result_route.append(time_route3)        #경로3 실시간추천

        return result_route

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


