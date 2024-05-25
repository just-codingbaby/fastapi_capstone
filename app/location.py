import os

import mysql.connector
from app import db_process

file_name = ['경부선_gps','중부선_gps']
directory = "/Users/jeonghaechan/projects/capstone-fastapi/data/"

connection = db_process.connect_to_mysql("location")        #location 스키마 연결

# 위도 경도 넣기
for name in file_name:
    files = sorted([file for file in os.listdir(directory+name) if not file.startswith('.DS_Store')])

    for file in files:
        # 파일 이름에서 정보 추출 (노드명, 위도, 경도)
        node_name, latitude, longitude = file[:-4].split("_")
        latitude = float(latitude)  # 위도
        longitude = float(longitude)  # 경도

        cursor = connection.cursor()
        query = f"INSERT INTO {name} (노드명, 위도, 경도) VALUES (%s, %s, %s)"
        val = (node_name, latitude, longitude)
        cursor.execute(query, val)
        connection.commit()
        cursor.close()

connection.close()

