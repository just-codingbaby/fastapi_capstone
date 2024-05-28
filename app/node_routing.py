from app import db_process

# (경도,위도) 리스트
loc_list = [[127.22031199549663,37.52381114479651],
            [127.24098253938524,37.4842955984355],
            [127.25374529878387,37.465931869766976],
            [127.26317194991778,37.45201993175246],
            [127.27938707326331,37.4451845824706],
            [127.3037457517851,37.41304840725884],
            [127.3153660401467,37.386699251295425],
            [127.3235283049389,37.346852000680954],
            [127.32537388142184,37.341697575159905],
            [127.34740718560963,37.31515596805733],

            [127.37471602201036,37.29412593065992],
            [127.39848711354917,37.27111247899416],
            [127.41987824596501,37.25030630765819],
            [127.43228336236245,37.21979076768787],
            [127.44276715850832,37.19452191600411],
            [127.44528074122488,37.10951014878058],
            [127.47493219534448,37.05570119062435],
            [127.47843298234658,37.001719008085665],
            [127.4669047573043,36.971587500531335],
            [127.474900100258,36.93254850659951],

           [127.47591573344967,36.89574803386027],
            [127.47905463319294,36.86141333965102],
            [127.49265031625535,36.828953862302306],
            [127.50766968194922,36.78987979374164],
            [127.44540352419638,36.718645828453454],

            [127.42458377120629,36.67464859084522],
            [127.4200886458462,36.60520675877221]]



def insert_into_table(cursor, node_name, latitude, longitude):
    query = "INSERT INTO tmp (노드명, 위도, 경도) VALUES (%s, %s, %s)"
    cursor.execute(query, (node_name, latitude, longitude))


connection_gps = db_process.connect_to_mysql("location")
cursor_gps = connection_gps.cursor()
connection_route = db_process.connect_to_mysql("경로")
cursor_route = connection_route.cursor()

for loc in loc_list:
    latitude = loc[1]   #위도
    longitude = loc[0] #경도

    query = "SELECT * FROM 중부선_gps WHERE 위도=%s AND 경도=%s"
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




