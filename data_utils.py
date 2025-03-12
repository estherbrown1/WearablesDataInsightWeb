import pandas as pd
import os
from datetime import datetime
import numpy as np
from google.cloud import storage
import re
from sql_utils import *
import io

# Define a function to clean the timestamp data later
def clean_timestamp_data(df):
    '''
    This function cleans the timestamp data in an input dataframe df
    Details are explained in the markdown chunk above
    '''
    
    # Calculate the final Unix timestamp (ms) by addressing the timezone difference
    df["unix_timestamp_cleaned"] = df["unixTimestampInMs"] + df["timezoneOffsetInMs"] # Retain this column for calculations
    df["timestamp_cleaned"] = df["unixTimestampInMs"] + df["timezoneOffsetInMs"]
    
    # Unix timestamp to human readable format 
    df["timestamp_cleaned"] = df["timestamp_cleaned"] / 1000.0 # Convert milliseconds to seconds
    df["timestamp_cleaned"] = df["timestamp_cleaned"].apply(lambda ts: datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')) # Convert to datetime object
    df['timestamp_cleaned'] = pd.to_datetime(df['timestamp_cleaned']) # Convert to datetime object 
    
    # Sort by 'timestamp_cleaned' (make sure the timestamps are in the correct order)
    df = df.sort_values(by = 'timestamp_cleaned')
    
    # Delete raw timestamp columns
    df = df.drop(columns = ["unixTimestampInMs", "timezoneOffsetInMs", "isoDate"])
    
    # Reorder the columns so that timestamp_cleaned is first
    columns = ['timestamp_cleaned'] + [col for col in df.columns if col != 'timestamp_cleaned']
    df = df[columns]
    
    return df


def get_csv_from_gcs(bucket_name, prefix, skiprows=5):
    """
    Function to list and read the first CSV file from a GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix (folder path) to list the CSV files from.
        skiprows (int): Number of rows to skip when reading the CSV.

    Returns:
        pd.DataFrame: The DataFrame of the first CSV file found.
    """
    # Initialize the GCP storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all CSV files in the specified path (prefix)
    blobs = bucket.list_blobs(prefix=prefix)
    # print(prefix)
    csv_files = [blob.name for blob in blobs if blob.name.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified path.")

    # Load the first CSV file found
    df_list = []
    non_empty = False
    for s in csv_files:
        csv_blob = bucket.blob(s)

        # Download the CSV content as a string and load it into a pandas DataFrame
        csv_content = csv_blob.download_as_text()
        # print(io.StringIO(csv_content))
        try:
            df = pd.read_csv(io.StringIO(csv_content), skiprows=skiprows)  # Using io.StringIO here
            df_list.append(df)
            non_empty = True
        except:
            pass
        
    assert non_empty 
    return df_list


# if __name__ == "__main__":
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "apcomp297-84a78a17c7a6.json" 
#     get_csv_from_gcs("physiological-data", "raw/Paige_3c9bbf15_labfront_export_09182024//garmin-connect-activity-move-iq-summary", skiprows=5)
    
def get_binary_indicator(labfront_exported_data_path, bucket_name = "physiological-data"):
    """
    Get a binary indicator for file types of interest in a GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        labfront_exported_data_path (str): The prefix (folder path) to check in the GCS bucket.

    Returns:
        numpy.ndarray: A binary indicator array for the presence of certain file types.
    """
    # Initialize the GCP storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all blobs (objects) in the specified folder path
    blobs = bucket.list_blobs(prefix=labfront_exported_data_path)

    # List of blob names (representing files or folders in GCS)
    blob_names = [blob.name for blob in blobs]

    # Binary indicator for file types of interest
    n_indicators = 13
    binary_indicator = np.zeros(n_indicators)

    # Iterate over the blobs (files/folders) to check for specific file types
    for i in range(n_indicators):
        for blob_name in blob_names:
            # Check for each specific type of file/folder in the blob names
            if "activity_details_summary" in blob_name:
                binary_indicator[i] = 1
            if "daily_summary" in blob_name:
                binary_indicator[i] = 1
            if "hrv_summary" in blob_name:
                binary_indicator[i] = 1
            if "sleep_summary" in blob_name:
                binary_indicator[i] = 1
            if "daily_heart_rate" in blob_name:
                binary_indicator[i] = 1
            if "hrv_values" in blob_name:
                binary_indicator[i] = 1
            if "respiration" in blob_name:
                binary_indicator[i] = 1
            if "sleep_respiration" in blob_name:
                binary_indicator[i] = 1
            if "sleep_stage" in blob_name:
                binary_indicator[i] = 1
            if "stress" in blob_name:
                binary_indicator[i] = 1
            if "epoch" in blob_name:
                binary_indicator[i] = 1
            if "bbi" in blob_name:
                binary_indicator[i] = 1
            if "step" in blob_name:
                binary_indicator[i] = 1

    # Convert to boolean array
    binary_indicator = binary_indicator.astype(bool)

    return binary_indicator


def clean_data(binary_indicator, labfront_exported_data_path, bucket_name = "physiological-data"):
    result = dict()
    if labfront_exported_data_path[-1] == "/": labfront_exported_data_path = labfront_exported_data_path[:-1]

    # If activity_details_summary was generated for the given Garmin watch 
    if binary_indicator[0]:
        # List all file(s) in the activity_details_summary directory -> read the .csv file(s) -> clean -> write  
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-activity-move-iq-summary") if file.endswith('.csv')]
        # activity_details_summary = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-activity-move-iq-summary/{csv_files[0]}", skiprows=5) 
        activity_details_summary = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-activity-move-iq-summary")
        activity_details_summary = clean_timestamp_data(activity_details_summary[0])
        result["activity_details_summary"] = activity_details_summary
        
    # If daily_summary was generated for the given Garmin watch 
    if binary_indicator[1]:
        # List all file(s) in the daily_summary directory -> read the .csv file(s) -> clean -> write
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-daily-summary") if file.endswith('.csv')]
        # daily_summary = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-daily-summary/{csv_files[0]}", skiprows=5) 
        daily_summary = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-daily-summary")
        daily_summary = clean_timestamp_data(daily_summary[0])
        result["daily_summary"] = daily_summary
        
    # If sleep_summary was generated for the given Garmin watch 
    if binary_indicator[3]:
        # List all file(s) in the sleep_summary directory -> read the .csv file(s) -> clean -> write
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-sleep-summary") if file.endswith('.csv')]
        # sleep_summary = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-sleep-summary/{csv_files[0]}", skiprows=5) 
        sleep_summary = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-sleep-summary")
        sleep_summary = clean_timestamp_data(sleep_summary[0])
        result["sleep_summary"] = sleep_summary
        
    # If daily_heart_rate was generated for the given Garmin watch 
    if binary_indicator[4]:
        # List all file(s) in the daily_heart_rate directory -> read the .csv file(s) -> clean -> write
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-daily-heart-rate") if file.endswith('.csv')]
        dataframes = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-daily-heart-rate", skiprows=5)
        # for file in csv_files:
        #     df = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-daily-heart-rate/{file}", skiprows = 5)
        #     df = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-sleep-summary")
        #     dataframes.append(df)
        daily_heart_rate = pd.concat(dataframes, ignore_index = True)
        daily_heart_rate = clean_timestamp_data(daily_heart_rate)
        daily_heart_rate = daily_heart_rate.loc[daily_heart_rate['beatsPerMinute'].shift() != daily_heart_rate['beatsPerMinute']]
        result["daily_heart_rate"] = daily_heart_rate
       
        
    # If respiration was generated for the given Garmin watch 
    if binary_indicator[6]:
        # List all file(s) in the respiration directory -> read the .csv file(s) -> clean -> write
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-respiration") if file.endswith('.csv')]
        dataframes = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-respiration", skiprows=5)
        # for file in csv_files:
        #     df = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-respiration/{file}", skiprows = 5)
        #     dataframes.append(df)
        respiration = pd.concat(dataframes, ignore_index = True)
        respiration = clean_timestamp_data(respiration)
        respiration = respiration[respiration['breathsPerMinute'] > 0]
        result["respiration"] = respiration
        
    # If sleep_respiration was generated for the given Garmin watch 
    if binary_indicator[7]:
        # List all file(s) in the respiration directory -> read the .csv file(s) -> clean -> write
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-sleep-respiration") if file.endswith('.csv')]
        dataframes = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-sleep-respiration", skiprows=5)
        # for file in csv_files:
        #     df = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-sleep-respiration/{file}", skiprows = 5)
        #     dataframes.append(df)
        sleep_respiration = pd.concat(dataframes, ignore_index = True)   
        sleep_respiration = clean_timestamp_data(sleep_respiration)
        result["sleep_respiration"] = sleep_respiration
        
    # If sleep_stage was generated for the given Garmin watch 
    if binary_indicator[8]:
        # List all file(s) in the sleep_stage directory -> read the .csv file(s) -> clean -> write
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-sleep-stage") if file.endswith('.csv')]
        # sleep_stage = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-sleep-stage/{csv_files[0]}", skiprows=5) 
        sleep_stage = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-sleep-stage", skiprows=5)
        sleep_stage = clean_timestamp_data(sleep_stage[0])
        result["sleep_stage"] = sleep_stage
        
    # If stress was generated for the given Garmin watch 
    if binary_indicator[9]:
        # List all file(s) in the stress directory -> read the .csv file(s) -> clean -> write
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-stress") if file.endswith('.csv')]
        dataframes = get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-stress", skiprows=5)
        # for file in csv_files:
        #     df = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-stress/{file}", skiprows = 5)
        #     dataframes.append(df)
        stress = pd.concat(dataframes, ignore_index = True)  
        stress = clean_timestamp_data(stress)
        stress = stress[stress['stressLevel'] > 0]
        result["stress"] = stress
        
    # If epoch was generated for the given Garmin watch 
    if binary_indicator[10]:
        # List all file(s) in the epoch directory -> read the .csv file(s) -> clean -> write
        # csv_files = [file for file in os.listdir(f"{labfront_exported_data_path}/garmin-connect-epoch") if file.endswith('.csv')]
        epoch  =  get_csv_from_gcs(bucket_name, f"{labfront_exported_data_path}/garmin-connect-stress", skiprows=5)
        # epoch = pd.read_csv(f"{labfront_exported_data_path}/garmin-connect-epoch/{csv_files[0]}", skiprows=5) 
        epoch = clean_timestamp_data(epoch[0])
        # epoch.to_csv(f"{labfront_exported_data_path}_cleaned/epoch.csv", index = False) 
        result["epoch"] = epoch
    
    return result 

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



def list_matching_directories(bucket_name, name_prefix):
    """
    List all directories that start with 'raw/{name_prefix}_' in the given GCP bucket.

    Args:
        bucket_name (str): The name of the GCP bucket.
        name_prefix (str): The name prefix to match, e.g., "Josha-Thomas".

    Returns:
        List of matching directories.
    """
    # Initialize the GCP storage client
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "apcomp297-84a78a17c7a6.json" 
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all blobs (objects) in the bucket
    blobs = bucket.list_blobs()

    # Create a regular expression pattern to match 'raw/{name_prefix}_'
    pattern = re.compile(rf'raw/{name_prefix}_.*?/')

    # Create a set to store unique directory names
    matching_directories = set()

    # Iterate through all the blobs (files)
    for blob in blobs:
        blob_name = blob.name

        # Check if the blob name matches the pattern
        match = pattern.match(blob_name)
        if match:
            # Extract the directory path
            directory = match.group()
            matching_directories.add(directory)  # Add it to the set to avoid duplicates

    return list(matching_directories)



def update_database(bucket_name="physiological-data"):
    """
    Update the MySQL (Amazon RDS) database with the user's labfront data.

    Connects to the RDS database, retrieves the `labfront_name` of users from the `users` table,
    and updates the database with their corresponding data from GCS.

    Args:
        bucket_name (str): The name of the GCS bucket.
    """
    # Connect to the RDS database
    conn = get_rds_connection()
    cursor = conn.cursor()

    # Get all labfront names from the "users" table
    cursor.execute("SELECT labfront_name FROM users")
    labfront_names = cursor.fetchall()  # This returns a list of tuples [(name1,), (name2,), ...]

    lname_to_uname = labfront_name_to_username()

    # Loop through each labfront name
    for labfront_name_tuple in labfront_names:
        labfront_name = labfront_name_tuple[0]  # Extract the actual name from the tuple
        
        # List the directories in GCS that match the current labfront name
        dirs = list_matching_directories(bucket_name, labfront_name)

        # Process each directory (assuming these directories represent different datasets)
        all = len(dirs)
        count = 0
        for labfront_exported_data_path in dirs:
            # Get the binary indicator for the available data in this directory
            binary_ind = get_binary_indicator(labfront_exported_data_path, bucket_name)

            # Clean the data from the available files
            df_dict = clean_data(binary_ind, labfront_exported_data_path, bucket_name)

            # Save the cleaned data into the database
            save_data(df_dict, lname_to_uname[labfront_name])
            count += 1
            print(f"{count} / {all}")

    # Close the connection
    conn.close()


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "apcomp297-84a78a17c7a6.json" 
    update_database(bucket_name="physiological-data")


