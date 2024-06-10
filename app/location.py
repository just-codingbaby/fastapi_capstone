import os
import mysql.connector
from app import db_process

file_name = ["경부선_VDE"]
directory = "/Users/jeonghaechan/projects/capstone-fastapi/data/"

connection = db_process.connect_to_mysql("location")  # location 스키마 연결

# 위도 경도 넣기
for name in file_name:
    files = sorted([file for file in os.listdir(directory + name) if not file.startswith('.DS_Store')])

    for file in files:
        # 파일 이름에서 정보 추출 (노드명, 위도, 경도)
        if file.endswith(".csv"):
            file_parts = file[:-4].split("_")
            if len(file_parts) == 3:
                node_name, latitude, longitude = file_parts
                try:
                    latitude = float(latitude)  # 위도
                    longitude = float(longitude)  # 경도
                except ValueError as e:
                    print(f"Error converting latitude or longitude to float in file {file}: {e}")
                    continue

                cursor = connection.cursor()
                query = f"INSERT IGNORE INTO {name} (노드명, 위도, 경도) VALUES (%s, %s, %s)"
                val = (node_name, latitude, longitude)
                try:
                    cursor.execute(query, val)
                    connection.commit()
                    print(f"Node Name: {node_name}, Latitude: {latitude}, Longitude: {longitude}")
                except mysql.connector.Error as e:
                    print(f"Error inserting data into table {name}: {e}")
                finally:
                    cursor.close()
            else:
                print(f"File name {file} does not match the expected format.")

connection.close()
