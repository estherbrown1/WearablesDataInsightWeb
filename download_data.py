import streamlit as st
import pandas as pd
import pymysql
from sql_utils import *

# Streamlit subpage
def download_user_data():
    st.title("Download User Data")
    
    # Input for admin to provide username
    user_name = st.text_input("Enter user name:")

    if user_name:
        # Connect to the RDS instance
        try:
            conn = get_rds_connection()
            with conn.cursor() as cursor:
                # Query to get all tables matching the pattern
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                
                # Extract relevant tables
                matching_tables = [table[0] for table in tables if table[0].startswith(f"{user_name}_")]
                
                if not matching_tables:
                    st.warning(f"No tables found for user '{user_name}'.")
                else:
                    # Display options for the admin to choose
                    y_options = [table[len(user_name) + 1:] for table in matching_tables]
                    selected_y = st.selectbox("Choose a table (Y):", y_options)
                    
                    if selected_y:
                        table_name = f"{user_name}_{selected_y}"
                        
                        # Button to download data as CSV
                        if st.button("Download CSV"):
                            try:
                                query = f"SELECT * FROM `{table_name}`"
                                df = pd.read_sql(query, conn)
                                csv = df.to_csv(index=False)
                                
                                st.download_button(
                                    label=f"Download {table_name}.csv",
                                    data=csv,
                                    file_name=f"{table_name}.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"Error fetching data from table {table_name}: {e}")
        except Exception as e:
            st.error(f"Database connection error: {e}")
        finally:
            conn.close()

