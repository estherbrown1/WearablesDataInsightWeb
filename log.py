# log_page.py
import streamlit as st
import sqlite3
from datetime import datetime
# Function to create/connect to a database and create the interventions table if it doesn't exist
def create_connection():
    conn = sqlite3.connect('interventions.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS interventions
                 (id INTEGER PRIMARY KEY,
                 intervention TEXT,
                 date TEXT,
                 start_time TEXT,
                 end_time TEXT);''')
    return conn
def add_intervention(conn, intervention, date, start_time, end_time):
    sql = '''INSERT INTO interventions(intervention, date, start_time, end_time)
             VALUES(?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, (intervention, date, start_time, end_time))
    conn.commit()
    return cur.lastrowid
def log_page():
    st.title('Log Interventions')
    with st.form("intervention_form"):
        intervention = st.selectbox('Select Intervention', ['Intervention 1', 'Intervention 2', 'Intervention 3'])
        date = st.date_input("Date", datetime.now())  # Add this line for date input
        start_time = st.time_input("Start Time")
        end_time = st.time_input("End Time")
        submitted = st.form_submit_button("Submit")
        if submitted:
            conn = create_connection()
            add_intervention(conn, intervention, date.strftime("%Y-%m-%d"), start_time.strftime("%H:%M"), end_time.strftime("%H:%M"))
            st.success("Intervention logged successfully")
if __name__ == '__main__':
    log_page()
