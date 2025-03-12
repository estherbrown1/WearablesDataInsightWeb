import streamlit as st
import pandas as pd
import zipfile
import os
from datetime import datetime
import pytz
from dateutil import rrule
import re
from sql_utils import get_rds_connection
from io import StringIO


def calendar_upload_page():
    st.title("Calendar Upload")
    
    uploaded_file = st.file_uploader("Upload your calendar ZIP file", type=['zip'])
    
    if uploaded_file is not None:
        try:
            # Process ZIP file in memory
            with zipfile.ZipFile(uploaded_file) as zip_ref:
                # Find first .ics file
                ics_files = [f for f in zip_ref.namelist() if f.endswith('.ics')]
                if not ics_files:
                    st.error("No ICS file found in ZIP")
                    return
                    
                # Read ICS file content
                ics_content = zip_ref.read(ics_files[0]).decode('utf-8')
                
                # Process ICS content to DataFrame
                df = process_ics_content(ics_content)
                
                # Preview the data
                st.write("Calendar Events Preview:")
                st.write(f"Total events: {len(df)}")
                st.dataframe(df.head())
                
                # Save button
                if st.button("Save to Database"):
                    try:
                        save_calendar_data(df, st.session_state.user)
                        st.success("Calendar data successfully saved!")
                    except Exception as e:
                        st.error(f"Error saving calendar data: {str(e)}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def process_ics_content(ics_content):
    """Process ICS content to DataFrame with recurring event expansion."""
    events = []
    event = {}
    inside_event = False
    timezone = pytz.timezone("America/New_York")
    
    for line in ics_content.splitlines():
        line = line.strip()
        if line.startswith("BEGIN:VEVENT"):
            inside_event = True
            event = {}
        elif line.startswith("END:VEVENT"):
            inside_event = False
            event.setdefault("Summary", "No Title")
            event.setdefault("Start Time", None)
            event.setdefault("End Time", None)
            event.setdefault("Location", None)
            event.setdefault("Description", None)
            event.setdefault("UID", str(datetime.now()))
            events.append(event.copy())
        elif inside_event:
            if line.startswith("SUMMARY:"):
                event["Summary"] = line.replace("SUMMARY:", "").strip()
            elif line.startswith("DTSTART"):
                match = re.search(r"(\d{8}T\d{6})", line)
                if match:
                    event["Start Time"] = datetime.strptime(
                        match.group(0), "%Y%m%dT%H%M%S"
                    ).astimezone(timezone)
            elif line.startswith("DTEND"):
                match = re.search(r"(\d{8}T\d{6})", line)
                if match:
                    event["End Time"] = datetime.strptime(
                        match.group(0), "%Y%m%dT%H%M%S"
                    ).astimezone(timezone)
            elif line.startswith("LOCATION:"):
                event["Location"] = line.replace("LOCATION:", "").strip()
            elif line.startswith("DESCRIPTION:"):
                event["Description"] = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("UID:"):
                event["UID"] = line.replace("UID:", "").strip()

    df = pd.DataFrame(events)
    return df

def save_calendar_data(df, user_name):
    """Save calendar data to the database."""
    conn = get_rds_connection()
    cursor = conn.cursor()
    
    try:
        # Create table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {user_name}_calendar_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                start_time DATETIME,
                end_time DATETIME NULL,
                summary VARCHAR(255),
                location TEXT NULL,
                description TEXT NULL,
                event_uid VARCHAR(255)
            )
        """)
        
        # Convert datetime columns to MySQL format
        df['Start Time'] = pd.to_datetime(df['Start Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['End Time'] = pd.to_datetime(df['End Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Insert events
        for _, row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO {user_name}_calendar_events 
                (start_time, end_time, summary, location, description, event_uid)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                row['Start Time'],
                row['End Time'] if pd.notna(row['End Time']) else None,
                str(row['Summary']) if pd.notna(row['Summary']) else 'No Title',
                str(row['Location']) if pd.notna(row['Location']) else None,
                str(row['Description']) if pd.notna(row['Description']) else None,
                str(row['UID'])
            ))
        
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()