import sqlite3
from tabulate import tabulate  # Make sure to install this using: pip install tabulate

# Connect to the database
conn = sqlite3.connect('traffic_violations.db')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in database:", tables)

# Use the first table (you can also directly write the table name here if known)
table_name = tables[0][0]

# Get column names
cursor.execute(f"PRAGMA table_info({table_name});")
columns_info = cursor.fetchall()
column_names = [col[1] for col in columns_info]

# Fetch first 10 rows
cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")
rows = cursor.fetchall()

# Print in table format
print("\nFirst 10 rows from table:", table_name)
print(tabulate(rows, headers=column_names, tablefmt="fancy_grid"))

# Close connection
conn.close()