import sqlite3
from tabulate import tabulate

def show_database_structure(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables in the database:")
    for table in tables:
        print(f"- {table[0]}")
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        
        # Prepare data for tabulate
        table_data = [[col[1], col[2], "PRIMARY KEY" if col[5] else ""] for col in columns]
        headers = ["Column Name", "Data Type", "Key"]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\n")

    # Sample data
    print("Sample data (first 5 rows):")
    cursor.execute(f"SELECT * FROM {tables[0][0]} LIMIT 5")
    sample_data = cursor.fetchall()
    headers = [description[0] for description in cursor.description]
    print(tabulate(sample_data, headers=headers, tablefmt="grid"))

    conn.close()

if __name__ == "__main__":
    db_path = '6_sql_rag/tesla_motors_data.db'
    show_database_structure(db_path)