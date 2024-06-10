from app import db_process

# (경도,위도) 리스트
loc_list = [
[127.10363403620896,37.31251422591168],
]



def insert_into_table(cursor, node_name, latitude, longitude):
    query = "INSERT INTO 역방향2 (노드명, 위도, 경도) VALUES (%s, %s, %s)"
    cursor.execute(query, (node_name, latitude, longitude))


connection_gps = db_process.connect_to_mysql("location")
cursor_gps = connection_gps.cursor()
connection_route = db_process.connect_to_mysql("경로")
cursor_route = connection_route.cursor()

for loc in loc_list:
    latitude = loc[1]   #위도
    longitude = loc[0] #경도

    query = "SELECT * FROM 경부선_VDE WHERE 위도=%s AND 경도=%s"
    cursor_gps.execute(query, (latitude, longitude))
    results = cursor_gps.fetchall()
    print(f"{results} / 위도: {latitude}/ 경도: {longitude}")

    for row in results:
        insert_into_table(cursor_route,row[0],row[1],row[2])

connection_route.commit()

cursor_route.close()
cursor_gps.close()
connection_route.close()
connection_gps.close()




