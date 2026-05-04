import os
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sql_utils import *

# Streamlit application
def data_upload():
    st.title("Physiological Data Uploader")
    st.write("Upload a ZIP file containing physiological data directories.")

    uploaded_file = st.file_uploader("Upload ZIP file", type=["zip"])
    if uploaded_file is not None:
        user_name = st.session_state.user
        st.write(f"Processing data for user: {user_name}")

        # Extract ZIP file
        temp_dir = extract_zip(uploaded_file)

        # List matching directories
        matching_dirs = get_matching_directories(temp_dir, user_name)
        if not matching_dirs:
            st.error("No matching directories found for the user.")
            shutil.rmtree(temp_dir)
            return

        progress = st.progress(0)
        for idx, labfront_exported_data_path in enumerate(matching_dirs):
            binary_ind = get_binary_indicator(labfront_exported_data_path)
            cleaned_data = clean_data(binary_ind, labfront_exported_data_path)
            save_data(cleaned_data, user_name)
            progress.progress((idx + 1) / len(matching_dirs))

        st.success("Database successfully updated!")
        shutil.rmtree(temp_dir)


def _detect_skiprows(csv_path: str) -> int:
    """
    Detect the correct number of rows to skip in a Labfront CSV so that
    the first row read is the column-header row.

    Labfront CSVs have either:
      - 4 metadata rows  (rows 0-3) then the header on row 4  → skiprows=4
      - 4 metadata rows + 1 blank row (rows 0-4) then header on row 5 → skiprows=5

    We detect this by checking whether row 4 (0-indexed) is blank or not.
    """
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [fh.readline() for _ in range(6)]
        # row 4 (index 4) is blank → header is on row 5
        if lines[4].strip() == "":
            return 5
        return 4
    except Exception:
        return 4  # safe default


def clean_timestamp_data(df: pd.DataFrame):
    '''
    This function cleans the timestamp data in an input dataframe df.
    Handles both timezoneOffsetInMs and timezone (string) columns.
    '''
    # Guard: if unixTimestampInMs is missing entirely, return df unchanged
    if "unixTimestampInMs" not in df.columns:
        return df

    if "timezoneOffsetInMs" in df.columns:
        df["unix_timestamp_cleaned"] = df["unixTimestampInMs"] + df["timezoneOffsetInMs"]
        df["timestamp_cleaned"] = df["unixTimestampInMs"] + df["timezoneOffsetInMs"]
        drop_cols = [c for c in ["unixTimestampInMs", "timezoneOffsetInMs", "isoDate"] if c in df.columns]
        df = df.drop(columns=drop_cols)
    else:
        df["unix_timestamp_cleaned"] = df["unixTimestampInMs"] + LOCAL_OFFSET
        df["timestamp_cleaned"] = df["unixTimestampInMs"] + LOCAL_OFFSET
        drop_cols = [c for c in ["unixTimestampInMs", "timezone", "isoDate"] if c in df.columns]
        df = df.drop(columns=drop_cols)

    # Unix timestamp (ms) → human-readable datetime
    df["timestamp_cleaned"] = df["timestamp_cleaned"] / 1000.0
    df["timestamp_cleaned"] = df["timestamp_cleaned"].apply(
        lambda ts: datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    )
    df['timestamp_cleaned'] = pd.to_datetime(df['timestamp_cleaned'])

    df = df.sort_values(by='timestamp_cleaned')

    # Reorder so timestamp_cleaned is first
    columns = ['timestamp_cleaned'] + [col for col in df.columns if col != 'timestamp_cleaned']
    df = df[columns]

    return df


# Utility functions for processing ZIP file
def extract_zip(zip_file_path):
    """
    Extracts the content of the ZIP file to a temporary directory.
    """
    temp_dir = "temp_extracted"
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        z.extractall(temp_dir)
    return temp_dir


def get_matching_directories(temp_dir, user_name):
    """
    List directories matching the user_name from the extracted ZIP file.
    """
    matching_dirs = []
    for folder in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder)
        if os.path.isdir(folder_path):
            matching_dirs.append(folder_path)
    return matching_dirs


# Core data cleaning and database update functions
def get_binary_indicator(labfront_exported_data_path):
    """
    Get a binary indicator for file types of interest in the local directories.
    """
    file_types = [
        "activity_details_summary",
        "daily_summary",
        "hrv_summary",
        "sleep_summary",
        "daily_heart_rate",
        "hrv_values",
        "respiration",
        "sleep_respiration",
        "sleep_stage",
        "stress",
        "epoch",
        "bbi",
        "step"
    ]
    binary_indicator = np.zeros(len(file_types))
    file_types_2 = [s.replace("_", "-") for s in file_types]

    for i, file_type in enumerate(file_types):
        if os.path.exists(os.path.join(labfront_exported_data_path, f"garmin-connect-{file_type}")):
            binary_indicator[i] = 1
        if os.path.exists(os.path.join(labfront_exported_data_path, f"garmin-connect-{file_types_2[i]}")):
            binary_indicator[i] = 1
        if os.path.exists(os.path.join(labfront_exported_data_path, f"garmin-device-{file_types_2[i]}")):
            binary_indicator[i] = 1

    return binary_indicator.astype(bool)


def get_csv_files_from_local(path):
    """
    Reads CSV files from a local directory, auto-detecting the correct
    number of header rows to skip for each file (4 or 5).

    Returns a list of pandas DataFrames.
    """
    csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    dfs = []
    for csv_file in csv_files:
        try:
            skiprows = _detect_skiprows(csv_file)
            df = pd.read_csv(csv_file, skiprows=skiprows)
            # Drop completely empty rows/columns that sometimes appear
            df = df.dropna(how='all')
            df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}")
    return dfs


def clean_data(binary_indicator, labfront_exported_data_path):
    """
    Clean data from the specified folder based on binary indicators.
    """
    result = {}

    file_mapping = {
        0: "activity_details_summary",
        1: "daily_summary",
        3: "sleep_summary",
        4: "daily_heart_rate",
        6: "respiration",
        7: "sleep_respiration",
        8: "sleep_stage",
        9: "stress",
        10: "epoch",
        11: "bbi",
        12: "step"
    }
    file_types_2 = {s: s.replace("_", "-") for _, s in file_mapping.items()}

    for idx, folder_name in file_mapping.items():
        if binary_indicator[idx]:
            folder_path = None
            if os.path.exists(os.path.join(labfront_exported_data_path, f"garmin-connect-{folder_name}")):
                folder_path = os.path.join(labfront_exported_data_path, f"garmin-connect-{folder_name}")
            elif os.path.exists(os.path.join(labfront_exported_data_path, f"garmin-connect-{file_types_2[folder_name]}")):
                folder_path = os.path.join(labfront_exported_data_path, f"garmin-connect-{file_types_2[folder_name]}")
            elif os.path.exists(os.path.join(labfront_exported_data_path, f"garmin-device-{file_types_2[folder_name]}")):
                folder_path = os.path.join(labfront_exported_data_path, f"garmin-device-{file_types_2[folder_name]}")

            if folder_path is None:
                continue

            dfs = get_csv_files_from_local(folder_path)
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                cleaned_df = clean_timestamp_data(combined_df)
                if "timestamp_cleaned" not in cleaned_df.columns:
                    # skip if timestamp cleaning failed (e.g. missing unixTimestampInMs)
                    continue
                if folder_name == "bbi":
                    result["rmssd"] = calculate_rmssd(cleaned_df)
                    cleaned_df = smooth_data(cleaned_df, folder_name, 180, 30)
                elif folder_name == "step":
                    if "steps" in cleaned_df.columns:
                        cleaned_df = cleaned_df[cleaned_df["steps"] != 0]
                result[folder_name] = cleaned_df

    return result


def calculate_rmssd(input_df: pd.DataFrame, window_seconds: int = 15*60,
                    step_seconds: int = int(0.5*60), threshold_seconds: int = 30*60,
                    min_valid_count: int = 20) -> pd.DataFrame:
    df = input_df.copy()
    df = df.set_index("timestamp_cleaned")
    df = df.sort_index()
    time_diff = df.index.to_series().diff(1).dt.total_seconds()
    bbi_diff_sq = df["bbi"].astype("float32").diff(1) ** 2

    bbi_diff_sq_sum = bbi_diff_sq.rolling(f"{window_seconds}s", min_periods=1).sum().where(time_diff <= threshold_seconds)
    bbi_diff_sq_count = bbi_diff_sq.rolling(f"{window_seconds}s", min_periods=1).count().where(time_diff <= threshold_seconds)

    df["bbi_diff_sq_sum"] = bbi_diff_sq_sum.where(time_diff <= threshold_seconds)
    df["bbi_diff_sq_count"] = bbi_diff_sq_count.where(time_diff <= threshold_seconds)

    agg_dict = {"bbi_diff_sq_sum": "sum", "bbi_diff_sq_count": "sum", "unix_timestamp_cleaned": "last"}
    if "deviceType" in df.columns:
        agg_dict["deviceType"] = "last"

    resampled_df = df.resample(f"{step_seconds}s").agg(agg_dict)
    resampled_df["rmssd"] = (resampled_df["bbi_diff_sq_sum"] / resampled_df["bbi_diff_sq_count"]) ** 0.5
    resampled_df["rmssd"] = resampled_df["rmssd"].where(resampled_df["bbi_diff_sq_count"] >= min_valid_count)
    resampled_df = resampled_df.reset_index()

    return resampled_df


def smooth_data(input_df: pd.DataFrame, var: str, window_seconds: int, step_seconds: int) -> pd.DataFrame:
    df = input_df.copy()
    df = df.set_index("timestamp_cleaned")
    df[f"{var}_rolling"] = df[var].rolling(f"{window_seconds}s", min_periods=1).mean()
    smoothed_df = df.resample(f"{step_seconds}s").last()
    smoothed_df = smoothed_df.reset_index()
    smoothed_df[var] = smoothed_df[f"{var}_rolling"]
    smoothed_df = smoothed_df.drop(columns=[f"{var}_rolling"])
    return smoothed_df


def save_data(df_dict, user_name):
    """
    Save data from the df_dict into a MySQL database (Amazon RDS), ensuring no duplicates.
    """
    conn = get_rds_connection()
    cursor = conn.cursor()

    for k in df_dict:
        df_dict[k]["name"] = user_name
        df_dict[k] = df_dict[k].map(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else x
        )
        df_dict[k] = df_dict[k].replace({pd.NA: None, np.nan: None})

    for df_name, df in df_dict.items():
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {user_name}_{df_name} ("
        for col in df.columns:
            col_type = pd_to_sql_type(df[col].dtype)
            create_table_sql += f"{col} {col_type}, "
        create_table_sql = create_table_sql.rstrip(', ') + ")"
        cursor.execute(create_table_sql)

        columns = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_sql = f"INSERT IGNORE INTO {user_name}_{df_name} ({columns}) VALUES ({placeholders})"
        data_to_insert = [tuple(row) for row in df.itertuples(index=False, name=None)]
        cursor.executemany(insert_sql, data_to_insert)

    conn.commit()
    conn.close()


def pd_to_sql_type(pd_type):
    if pd_type == 'int64':
        return 'BIGINT'
    elif pd_type == 'float64':
        return 'REAL'
    elif pd_type == 'bool':
        return 'BOOLEAN'
    else:
        return 'TEXT'


# Main Function
def process_local_zip(zip_file_path, user_name):
    """
    Process the data from a local ZIP file and update the database.
    """
    temp_dir = extract_zip(zip_file_path)
    matching_dirs = get_matching_directories(temp_dir, user_name)
    if not matching_dirs:
        print("No matching directories found for the user.")
        return

    for idx, labfront_exported_data_path in enumerate(matching_dirs):
        binary_ind = get_binary_indicator(labfront_exported_data_path)
        cleaned_data = clean_data(binary_ind, labfront_exported_data_path)
        print(labfront_exported_data_path)
        print(f"Processed {idx + 1} / {len(matching_dirs)} directories")

    print("Database successfully updated!")



