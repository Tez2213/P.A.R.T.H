import psycopg2

conn = psycopg2.connect(
    dbname="parthdb",
    user="parth_user",
    password="Owlfitter",
    host="localhost"
)

cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())
conn.close()
