import psycopg2
import csv

conn = psycopg2.connect("postgresql://postgres:postgres@postgres:5432/postgres")

with conn.cursor() as cur:
    cur.execute("SELECT * FROM conversation_memory ORDER BY turn_id")
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]

with open("conversation_memory.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(cols)
    writer.writerows(rows)
