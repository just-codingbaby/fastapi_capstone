import os

import mysql.connector
import pandas as pd

# MySQL 서버에 연결
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Wjdgocks2!',
    database='경부선'
)
print("연결 성공")

directory = "/Users/jeonghaechan/projects/capstone-fastapi/data/경부선_gps"
files = sorted([file for file in os.listdir(directory) if not file.startswith('.DS_Store')])


for file in files:
    # 파일 이름에서 정보 추출 (노드명, 위도, 경도)
    node_name, latitude, longitude = file[:-4].split("_")
    latitude = float(latitude)  # 위도
    longitude = float(longitude)  # 경도

    cursor = connection.cursor()

    query = "INSERT INTO location (노드명, 위도, 경도) VALUES (%s, %s, %s)"
    val = (node_name, latitude, longitude)
    cursor.execute(query, val)
    connection.commit()
    cursor.close()

# 연결 닫기
connection.close()
