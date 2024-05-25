import mysql.connector

# MySQL 연결 정보
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "Wjdgocks2!"
          #추후 변동예정

# MySQL 연결 함수
def connect_to_mysql(line):
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=line
    )

def get_node(line):
    db_connection = connect_to_mysql(line)
    cursor = db_connection.cursor()

    # '노드명' 열의 값을 가져오는 쿼리
    query = "SELECT 노드명 FROM location"
    cursor.execute(query)
    node_names = [row[0] for row in cursor.fetchall()]  # 모든 결과를 리스트에 담음

    cursor.close()
    db_connection.close()

    return node_names

def get_location(node_name: str, line):
    db_connection = connect_to_mysql(line)
    cursor = db_connection.cursor()

    # 특정 노드명을 가진 행을 쿼리
    query = "SELECT 위도, 경도 FROM location WHERE 노드명 = %s"
    cursor.execute(query, (node_name,))
    result = cursor.fetchone()
    result = (float(result[0]), float(result[1]))

    cursor.close()
    db_connection.close()


    return result

#
# # 노드명들 다 불러옴
# nodes = get_node()
#
# result = []
# for node in nodes:
#     if node < '0010VDS01600' or node > '0010VDS11200':
#         continue
#
#     print(node)
#     result.append(get_location(node))
#
# # Demical 타입을 float로
# # result = [(float(row[0]), float(row[1])) for row in result]
# print(result)