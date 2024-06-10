import os
import pandas as pd
import mysql.connector
from mysql.connector import errorcode

# MySQL 연결 설정
config = {
    'user': 'root',
    'password': 'Wjdgocks2!',
    'host': 'localhost',
    'database': '중부선_VDE',
    'raise_on_warnings': True
}

# 연결 생성
try:
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    print("MySQL에 연결되었습니다.")
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("사용자 이름 또는 비밀번호가 잘못되었습니다.")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("데이터베이스가 존재하지 않습니다.")
    else:
        print(err)
    exit()

# 특정 디렉토리 경로 설정
directory_path = '/Users/jeonghaechan/projects/capstone-fastapi/data/중부선_VDE'
files = sorted([file for file in os.listdir(directory_path) if not file.startswith('.DS_Store')])

# 디렉토리 내 모든 CSV 파일 목록 가져오기
for filename in files:
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)

        # CSV 파일을 pandas DataFrame으로 읽기
        df = pd.read_csv(file_path)

        # 테이블 이름 설정 (파일명에서 확장자 제거)
        table_name = os.path.splitext(filename)[0]

        # 테이블 생성 쿼리 작성
        columns = df.columns
        column_definitions = ', '.join([f'`{col}` TEXT' for col in columns])
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            {column_definitions}
        )
        """

        # 테이블 생성
        try:
            cursor.execute(create_table_query)
        except mysql.connector.Error as err:
            print(f"테이블 생성 실패: {err}")
            continue

        # DataFrame의 데이터를 테이블에 삽입
        for _, row in df.iterrows():
            placeholders = ', '.join(['%s'] * len(row))
            insert_query = f"INSERT INTO `{table_name}` VALUES ({placeholders})"
            try:
                cursor.execute(insert_query, tuple(row))
            except mysql.connector.Error as err:
                print(f"데이터 삽입 실패: {err}")
                continue

        # 변경사항 커밋
        cnx.commit()
        print(f'{table_name} 테이블이 {config["database"]} 데이터베이스에 저장되었습니다.')

# 연결 종료
cursor.close()
cnx.close()
print('모든 CSV 파일이 데이터베이스에 저장되었습니다.')
