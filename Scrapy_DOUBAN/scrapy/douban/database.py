import pymysql

# 以下根据本地环境自己填
MYSQL_DB = ''
MYSQL_USER = ''
MYSQL_PASS = ''
MYSQL_HOST = ''  # 'localhost'

connection = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER,
                             password=MYSQL_PASS, db=MYSQL_DB,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
