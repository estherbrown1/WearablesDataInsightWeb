import pymysql

def get_rds_connection():
    return pymysql.connect(
        host="apcomp297.chg8skogwghf.us-east-2.rds.amazonaws.com",
            user="yilinw",
            password="wearable42",
            database="wearable",
            port=3306
        )

def check_g_study_data_tables():
    conn = get_rds_connection()
    cursor = conn.cursor()

    # Get all g_study_data tables
    cursor.execute("SHOW TABLES LIKE 'g\\_study\\_data%'")
    tables = cursor.fetchall()

    print(f"Found {len(tables)} g_study_data tables:")
    for i, table in enumerate(tables):
        print(f"{i+1}. {table[0]}")

    print("\nSelect a table number to view structure (or 0 to see events/interventions tables):")
    try:
        choice = int(input("> "))
        if choice == 0:
            # Show events and interventions tables structure
            print("\n===== EVENTS TABLE =====")
            cursor.execute("DESCRIBE events")
            columns = cursor.fetchall()
            for col in columns:
                print(f"- {col[0]}: {col[1]}")

            print("\n===== INTERVENTIONS TABLE =====")
            cursor.execute("DESCRIBE interventions")
            columns = cursor.fetchall()
            for col in columns:
                print(f"- {col[0]}: {col[1]}")
        elif 1 <= choice <= len(tables):
            table_name = tables[choice-1][0]
            print(f"\nStructure for table: {table_name}")

            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            for col in columns:
                print(f"- {col[0]}: {col[1]}")

            cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
            rows = cursor.fetchone()
            if rows:
                print("\nSample row:")
                for i, col in enumerate(columns):
                    if i < len(rows):
                        print(f"  {col[0]}: {rows[i]}")
        else:
            print("Invalid choice.")
    except ValueError:
        print("Please enter a valid number.")

    conn.close()

if __name__ == "__main__":
    try:
        check_g_study_data_tables()
    except Exception as e:
        print(f"Error: {e}")
# Removed the EOF line that was here