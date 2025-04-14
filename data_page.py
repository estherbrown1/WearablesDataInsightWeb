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

def clean_timestamp_data(df: pd.DataFrame):
    '''
    This function cleans the timestamp data in an input dataframe df
    Details are explained in the markdown chunk above
    '''
    if "timezoneOffsetInMs" in df.columns:
        # Calculate the final Unix timestamp (ms) by addressing the timezone difference
        df["unix_timestamp_cleaned"] = df["unixTimestampInMs"] + df["timezoneOffsetInMs"] # Retain this column for calculations
        df["timestamp_cleaned"] = df["unixTimestampInMs"] + df["timezoneOffsetInMs"]
        
        # Delete raw timestamp columns
        df = df.drop(columns = ["unixTimestampInMs", "timezoneOffsetInMs", "isoDate"])
    else:
        df["unix_timestamp_cleaned"] = df["unixTimestampInMs"] + LOCAL_OFFSET
        df["timestamp_cleaned"] = df["unixTimestampInMs"] + LOCAL_OFFSET
        
        # Delete raw timestamp columns
        df = df.drop(columns = ["unixTimestampInMs", "timezone", "isoDate"])
        
    # Unix timestamp to human readable format 
    df["timestamp_cleaned"] = df["timestamp_cleaned"] / 1000.0 # Convert milliseconds to seconds
    df["timestamp_cleaned"] = df["timestamp_cleaned"].apply(lambda ts: datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')) # Convert to datetime object
    df['timestamp_cleaned'] = pd.to_datetime(df['timestamp_cleaned']) # Convert to datetime object 
    
    # Sort by 'timestamp_cleaned' (make sure the timestamps are in the correct order)
    df = df.sort_values(by = 'timestamp_cleaned')
        
    # Reorder the columns so that timestamp_cleaned is first
    columns = ['timestamp_cleaned'] + [col for col in df.columns if col != 'timestamp_cleaned']
    df = df[columns]
        
    return df


# Utility functions for processing ZIP file
def extract_zip(zip_file_path):
    """
    Extracts the content of the ZIP file to a temporary directory.

    Args:
        zip_file_path (str): Path to the ZIP file.

    Returns:
        temp_dir: Path to the extracted directory.
    """
    temp_dir = "temp_extracted"
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        z.extractall(temp_dir)
    return temp_dir


def get_matching_directories(temp_dir, user_name):
    """
    List directories matching the user_name from the extracted ZIP file.

    Args:
        temp_dir (str): Path to the temporary extracted directory.
        user_name (str): Name of the user to match directories for.

    Returns:
        List of matching directories.
    """
    matching_dirs = []
    for folder in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            matching_dirs.append(folder_path)
    return matching_dirs

# Core data cleaning and database update functions
def get_binary_indicator(labfront_exported_data_path):
    """
    Get a binary indicator for file types of interest in the local directories.

    Args:
        labfront_exported_data_path (str): The path to the local folder to check.

    Returns:
        numpy.ndarray: A binary indicator array for the presence of certain file types.
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


def get_csv_files_from_local(path, skiprows=5):
    """
    Reads CSV files from a local directory and returns a list of DataFrames.

    Args:
        path (str): The local path to the directory containing CSV files.
        skiprows (int): Number of rows to skip while reading the CSV.

    Returns:
        List of pandas DataFrames.
    """
    csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    dfs = []
    for csv_file in csv_files:
        try:
            dfs.append(pd.read_csv(csv_file, skiprows=skiprows))
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}")
    return dfs


def clean_data(binary_indicator, labfront_exported_data_path):
    """
    Clean data from the specified folder based on binary indicators.

    Args:
        binary_indicator (np.ndarray): Binary array indicating the available file types.
        labfront_exported_data_path (str): Path to the folder containing raw data.

    Returns:
        A dictionary of cleaned DataFrames.
    """
    result = {}

    # Map binary indicators to specific data cleaning operations
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
            if os.path.exists(os.path.join(labfront_exported_data_path, f"garmin-connect-{folder_name}")):
                folder_path = os.path.join(labfront_exported_data_path, f"garmin-connect-{folder_name}")
            elif os.path.exists( os.path.join(labfront_exported_data_path, f"garmin-connect-{file_types_2[folder_name]}")):
                folder_path = os.path.join(labfront_exported_data_path, f"garmin-connect-{file_types_2[folder_name]}")
            elif os.path.exists(os.path.join(labfront_exported_data_path, f"garmin-device-{file_types_2[folder_name]}")):
                folder_path = os.path.join(labfront_exported_data_path, f"garmin-device-{file_types_2[folder_name]}")
            
            dfs = get_csv_files_from_local(folder_path)
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                cleaned_df = clean_timestamp_data(combined_df)
                if folder_name == "bbi":
                    result["rmssd"] = calculate_rmssd(cleaned_df) # Add RMSSD data if BBI data is present
                    cleaned_df = smooth_data(cleaned_df, folder_name, 180, 90) # Smooth BBI data
                elif folder_name == "step":
                    cleaned_df = cleaned_df[cleaned_df["steps"] != 0] # Drop all records where # of steps did not change
                result[folder_name] = cleaned_df

    return result

def calculate_rmssd(input_df:pd.DataFrame, window_seconds:int=15*60, step_seconds:int=5*60, threshold_seconds:int=30*60, min_valid_count:int=20) -> pd.DataFrame:
    # Make a copy of the dataframe to avoid changing the original
    df = input_df.copy()

    # Set cleaned timeframe as index
    df = df.set_index("timestamp_cleaned")

    # Compute squared distance
    df = df.sort_index()
    time_diff = df.index.to_series().diff(1).dt.total_seconds()
    bbi_diff_sq = df["bbi"].astype("float32").diff(1) ** 2

    # Get sum and counts of squared distance
    bbi_diff_sq_sum = bbi_diff_sq.rolling(f"{window_seconds}s", min_periods=1).sum().where(time_diff <= threshold_seconds)
    bbi_diff_sq_count = bbi_diff_sq.rolling(f"{window_seconds}s", min_periods=1).count().where(time_diff <= threshold_seconds)

    # Create new column only for valid intervals
    df["bbi_diff_sq_sum"] = bbi_diff_sq_sum.where(time_diff <= threshold_seconds)
    df["bbi_diff_sq_count"] = bbi_diff_sq_count.where(time_diff <= threshold_seconds)

    # Resample to get RMSSD in specific step intervals
    resampled_df = df.resample(f"{step_seconds}s").agg({"bbi_diff_sq_sum":"sum", "bbi_diff_sq_count":"sum","deviceType":"last","unix_timestamp_cleaned":"last"})

    # Calculate mean of squared distance and take square root to get RMSSD
    resampled_df["rmssd"] = (resampled_df["bbi_diff_sq_sum"] / resampled_df["bbi_diff_sq_count"]) ** 0.5
    resampled_df["rmssd"] = resampled_df["rmssd"].where(resampled_df["bbi_diff_sq_count"] >= min_valid_count)

    # Reset index, drop intermediary columns and return the dataframe
    resampled_df = resampled_df.reset_index()

    return resampled_df

def smooth_data(input_df:pd.DataFrame, var:str, window_seconds:int, step_seconds:int) -> pd.DataFrame:
    """
    Smooths the data in the dataframe using the specified window or step.

    Args:
        input_df (pd.DataFrame): Dataframe to smooth.
        var (str): Column name to smooth
        window_seconds (int): Window size of time to smooth in seconds.
        step_seconds (int): Time step for each window in seconds.

    Returns:
        pd.DataFrame of smoothed data
    """
    # Make a copy of the dataframe to avoid changing the original
    df = input_df.copy()

    # Set cleaned timeframe as index
    df = df.set_index("timestamp_cleaned") 

    # Compute rolling mean
    df[f"{var}_rolling"] = df[var].rolling(f"{window_seconds}s", min_periods=1).mean()

    # Resample and take the last available value within each 90s bin
    smoothed_df = df.resample(f"{step_seconds}s").last()
    smoothed_df = smoothed_df.reset_index()

    # Set variable column to rolling column and drop rolling column 
    smoothed_df[var] = smoothed_df[f"{var}_rolling"]
    smoothed_df = smoothed_df.drop(columns=[f"{var}_rolling"])

    return smoothed_df 

def save_data(df_dict, user_name):
    """
    Save data from the df_dict into a MySQL database (Amazon RDS), ensuring no duplicates.
    
    Args:
        df_dict: A dictionary where keys are table names and values are pandas DataFrames.
        user_name: The name of the user to associate with the data.
    """
    # Connect to the RDS database
    conn = get_rds_connection()
    cursor = conn.cursor()

    # Format DataFrames for SQL and prepare for insertion
    for k in df_dict:
        df_dict[k]["name"] = user_name
        df_dict[k] = df_dict[k].applymap(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else x)
        df_dict[k] = df_dict[k].replace({pd.NA: None, np.nan: None})  # Replace NaN with None for MySQL compatibility

    for df_name, df in df_dict.items():
        # Generate the table creation SQL with columns matching the DataFrame structure
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {user_name}_{df_name} ("
        for col in df.columns:
            col_type = pd_to_sql_type(df[col].dtype)
            create_table_sql += f"{col} {col_type}, "
        create_table_sql = create_table_sql.rstrip(', ') + ")"
        cursor.execute(create_table_sql)

        # Prepare bulk insert data
        columns = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_sql = f"INSERT IGNORE INTO {user_name}_{df_name} ({columns}) VALUES ({placeholders})"
        
        # Convert DataFrame to list of tuples and execute bulk insert
        data_to_insert = [tuple(row) for row in df.itertuples(index=False, name=None)]
        cursor.executemany(insert_sql, data_to_insert)

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()



def pd_to_sql_type(pd_type):
    """
    Convert pandas data types to SQLite data types.
    
    Args:
        pd_type: pandas dtype
    
    Returns:
        A string representing the corresponding SQLite data type.
    """
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

    Args:
        zip_file_path (str): Path to the ZIP file.
        user_name (str): The name of the user to process data for.
    """
    # Extract ZIP file
    temp_dir = extract_zip(zip_file_path)

    # List matching directories
    matching_dirs = get_matching_directories(temp_dir, user_name)
    if not matching_dirs:
        print("No matching directories found for the user.")
        return
    

    # Update database
    for idx, labfront_exported_data_path in enumerate(matching_dirs):
        binary_ind = get_binary_indicator(labfront_exported_data_path)
        cleaned_data = clean_data(binary_ind, labfront_exported_data_path)
        # save_data(cleaned_data, user_name)
        print(labfront_exported_data_path)
        print(f"Processed {idx + 1} / {len(matching_dirs)} directories")

    print("Database successfully updated!")




