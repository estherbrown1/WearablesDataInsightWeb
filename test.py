# from google.cloud import storage

# import streamlit as st
# from data_utils import update_database
# import pymysql



# def list_all_files_in_bucket(bucket_name):
#     """
#     Lists all the files in a given Google Cloud Storage bucket.

#     Args:
#         bucket_name (str): The name of the GCS bucket.
        
#     Returns:
#         List of file paths in the bucket.
#     """
#     # Initialize the GCS client
#     client = storage.Client()

#     # Get the bucket
#     bucket = client.bucket(bucket_name)

#     # List all blobs (files) in the bucket
#     blobs = bucket.list_blobs()

#     # Collect file names (file paths)
#     file_list = [blob.name for blob in blobs]

#     return file_list

# if __name__ == "__main__":
#     # Example usage:
#     # bucket_name="physiological-data"
#     # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "apcomp297-84a78a17c7a6.json" 
#     # files = list_all_files_in_bucket(bucket_name)
#     # for file in files:
#     #     print(file)
#     conn = pymysql.connect(
#     host="apcomp297.chg8skogwghf.us-east-2.rds.amazonaws.com",
#     user="yilinw",
#     password="wearable42",
#     # database="wearable",
#     port=3306  # MySQL default port
#     )
#     cursor = conn.cursor()
#     # cursor.execute("DROP DATABASE IF EXISTS wearable;")
#     # conn.commit()

#     # try:
#     #     cursor = conn.cursor()
#     #     print("*" * 20)

#     #     # Switch to the 'wearable' database
#     #     cursor.execute("USE wearable;")
#     #     print("Switched to database 'wearable'.")

#     #     # Get all tables that start with 'j_'
#     #     cursor.execute("SHOW TABLES LIKE 'j\\_%';")
#     #     tables = cursor.fetchall()

#     #     # Loop through the tables and drop each one
#     #     for (table_name,) in tables:
#     #         drop_query = f"DROP TABLE `{table_name}`;"
#     #         cursor.execute(drop_query)
#     #         print(f"Dropped table: {table_name}")

#     #     conn.commit()  # Commit changes

#     # except pymysql.MySQLError as e:
#     #     print("Error while accessing MySQL:", e)

#     #     cursor.execute("SHOW DATABASES;")
#     #     databases = cursor.fetchall()

#     # # Print or display the list of databases
#     # for db in databases:
#     #     print(db[0])
#     conn.close()
