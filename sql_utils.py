import pandas as pd
import numpy as np
import pymysql
import time

import pymysql.cursors

if time.localtime().tm_isdst and time.daylight:
    LOCAL_OFFSET = -time.altzone * 1000  # Adjust for DST
else:
    LOCAL_OFFSET = -time.timezone * 1000  # Standard time offset

def get_rds_connection():
    return pymysql.connect(
        host="apcomp297.chg8skogwghf.us-east-2.rds.amazonaws.com",  # Your RDS endpoint
        user="yilinw",  # Your RDS username
        password="wearable42",  # Your RDS password
        database="wearable",  # Database name on RDS
        port=3306
    )

def get_df_from_query(query:str, conn:pymysql.connections.Connection) -> pd.DataFrame:
    """Helper function that returns the results of a SQL query as a dataframe."""
    with conn.cursor() as cursor:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
    return df

def check_user_credentials(username, password=None):
    """
    Check if a user exists and if the password matches (if provided).
    Returns True if credentials are valid, False otherwise.
    """
    conn = get_rds_connection()
    cursor = conn.cursor()
    
    try:
        # Check if password column exists
        cursor.execute("DESCRIBE users")
        columns = [column[0] for column in cursor.fetchall()]
        
        if 'password' in columns and password is not None:
            # Check username and password
            cursor.execute("SELECT * FROM users WHERE name = %s AND password = %s", 
                          (username, password))
        else:
            # Check only username
            cursor.execute("SELECT * FROM users WHERE name = %s", (username,))
        
        result = cursor.fetchone()
        return result is not None
    except Exception as e:
        print(f"Error checking credentials: {e}")
        return False
    finally:
        conn.close()

def fetch_past_options(user_name, var_name="events"):
    conn = get_rds_connection()
    cursor = conn.cursor()

    # Fetch distinct options based on the user name from the RDS database
    cursor.execute(f"SELECT {var_name} FROM users WHERE name = %s", (user_name,))
    past_responses = cursor.fetchall()

    conn.close()

    # Convert the results into a flat list of strings
    options = []
    if past_responses and past_responses[0][0]:  # Check if past responses exist and are not None
        options = past_responses[0][0].split("|||")  # Split on delimiter used in your database

    return options

def save_other_response(user_name, response, var_name="events"):
    conn = get_rds_connection()
    cursor = conn.cursor()

    # Fetch existing responses from the specified column for the user
    cursor.execute(f"SELECT {var_name} FROM users WHERE name = %s", (user_name,))
    past_responses = cursor.fetchall()

    # Convert the results into a flat list of strings
    options = []
    if past_responses and past_responses[0][0]:  # Check if past responses exist and are not None
        options = past_responses[0][0].split("|||")
    
    options.append(response)
    new_options = "|||".join(options)

    # Update the user's response in the RDS database
    cursor.execute(f"UPDATE users SET {var_name} = %s WHERE name = %s", (new_options, user_name))
    
    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

def record_event_in_database(user, start_time, end_time, event_type, var_name):
    """
    Records an event (intervention/activity) in the database.
    
    Args:
        user (str): Username
        start_time (datetime): Start time of the event
        end_time (datetime): End time of the event
        event_type (str): Type of event/intervention
        var_name (str): Name of the table ('interventions' or other)
    """
    conn = get_rds_connection()
    cursor = conn.cursor()

    try:
        # First check if the table exists
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {var_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                start_time DATETIME,
                end_time DATETIME,
                {var_name} VARCHAR(255)
            )
        """)
        conn.commit()

        # Insert the event details into the specified table
        cursor.execute(f'''
            INSERT INTO {var_name} (name, start_time, end_time, {var_name})
            VALUES (%s, %s, %s, %s)
        ''', (user, start_time, end_time, event_type))

        conn.commit()
    except Exception as e:
        print(f"Error recording event: {e}")
        conn.rollback()
    finally:
        conn.close()

# Add a function to create or update the users table
def ensure_users_table():
    """
    Ensure the users table exists with the required columns.
    """
    conn = get_rds_connection()
    cursor = conn.cursor()
    
    try:
        # Check if the users table exists
        cursor.execute("SHOW TABLES LIKE 'users'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            # Create the users table with name and password columns
            cursor.execute('''
                CREATE TABLE users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) UNIQUE,
                    password VARCHAR(255),
                    events TEXT,
                    interventions TEXT,
                    constraints TEXT,
                    labfront_name VARCHAR(255)
                )
            ''')
            print("Created users table")
        else:
            # Check if password column exists
            cursor.execute("DESCRIBE users")
            columns = [column[0] for column in cursor.fetchall()]
            
            # Add password column if it doesn't exist
            if 'password' not in columns:
                cursor.execute("ALTER TABLE users ADD password VARCHAR(255)")
                print("Added password column to users table")
        
        conn.commit()
    except Exception as e:
        print(f"Error ensuring users table: {e}")
        conn.rollback()
    finally:
        conn.close()

# Add a function to register a new user
def register_user(username, password=None):
    """
    Register a new user in the database.
    """
    # First ensure the table exists with the right structure
    ensure_users_table()
    
    conn = get_rds_connection()
    cursor = conn.cursor()
    
    try:
        # Check if password column exists
        cursor.execute("DESCRIBE users")
        columns = [column[0] for column in cursor.fetchall()]
        
        if 'password' in columns and password is not None:
            # Insert with password
            cursor.execute('''
                INSERT INTO users (name, password)
                VALUES (%s, %s)
            ''', (username, password))
        else:
            # Insert without password
            cursor.execute('''
                INSERT INTO users (name)
                VALUES (%s)
            ''', (username,))
        
        conn.commit()
        return True
    except pymysql.err.IntegrityError:
        # User already exists
        return False
    except Exception as e:
        print(f"Error registering user: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()



import pandas as pd
import numpy as np
import pymysql
import time

def fetch_data(var, var_dict, user_name):
    """
    The `fetch_data` function connects to an RDS database to retrieve data from a specific table based
    on user input and variable dictionary mapping, returning the data as a pandas DataFrame with error
    handling for various scenarios.
    
    :param var: The `var` parameter is a string representing a variable key that is used to look up a
    corresponding table name in the `var_dict` dictionary
    :param var_dict: The `var_dict` parameter is a dictionary where the keys are strings and the values
    are also strings. It is used to map a variable (`var`) to a specific value that helps determine the
    table name from which data needs to be fetched. The function `fetch_data` uses this dictionary to
    construct
    :param user_name: The `user_name` parameter in the `fetch_data` function represents the name of the
    user whose data you want to fetch from the RDS database. This user name is used to construct the
    table name by combining it with the value corresponding to `var` in the `var_dict` dictionary
    :return: The function `fetch_data` returns a pandas DataFrame containing the selected data from the
    table corresponding to the provided `var` and `user_name`. If an error occurs during the data
    retrieval process, it returns an empty DataFrame with a user-friendly error message indicating the
    issue encountered.
    """
    '''
    var: str
    var_dict: dict : str -> str
    Connects to RDS database to fetch all data from the table
    corresponding to user_name and var_dict[var].
    Returns the selected data as a pandas DataFrame.
    '''
    
    # Ensure the variable exists in the dictionary
    if var not in var_dict:
        raise ValueError(f"Variable '{var}' not found in var_dict.")
    
    # Get the table name from var_dict
    table_name = f"{user_name}_{var_dict[var]}"
    
    # Open connection to the RDS database
    conn = get_rds_connection()
    
    try:
        # SQL query to select data from the specified table
        query = f"SELECT * FROM {table_name}"
        
        # Execute the query and load data into a pandas DataFrame
        df = get_df_from_query(query, conn)
        return df
    
    except (pymysql.err.ProgrammingError, pymysql.err.OperationalError) as e:
        # Handle the case where the table doesn't exist
        if "doesn't exist" in str(e):
            # Return an empty DataFrame with a user-friendly error message
            empty_df = pd.DataFrame()
            empty_df.attrs['error_message'] = f"No data available for {var}. Please add data before visualizing."
            return empty_df
        else:
            # Handle other database errors with a user-friendly message
            empty_df = pd.DataFrame()
            empty_df.attrs['error_message'] = f"Database error: Unable to access data for {var}."
            return empty_df
    except Exception as e:
        # Handle other exceptions with a generic message
        empty_df = pd.DataFrame()
        empty_df.attrs['error_message'] = f"Unable to load data. Please verify that you have uploaded data and try again."
        return empty_df
    finally:
        # Close the connection
        conn.close()

def labfront_name_to_username():
    '''
    Connects to the RDS database to select name and labfront_name from the users table.
    Returns a dictionary d such that d[labfront_name] = name.
    '''
    
    # Open connection to the RDS database
    conn = get_rds_connection()
    
    # SQL query to fetch name and labfront_name from users table
    query = "SELECT name, labfront_name FROM users"
    
    # Execute the query and fetch all results
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Close the connection
    conn.close()
    
    # Create a dictionary where labfront_name is the key and name is the value
    labfront_dict = {labfront_name: name for name, labfront_name in rows}
    
    return labfront_dict

def validate_var_dict(var_dict):
    """
    Validates that the variable dictionary has the expected structure and mappings.
    Returns True if valid, False otherwise.
    """
    try:
        # Check if var_dict is a dictionary
        if not isinstance(var_dict, dict):
            print(f"ERROR: var_dict is not a dictionary, it's a {type(var_dict)}")
            return False
            
        # Check if var_dict has entries
        if not var_dict:
            print("ERROR: var_dict is empty")
            return False
            
        # Print all entries in var_dict for debugging
        print("DEBUG: var_dict contents:")
        for key, value in var_dict.items():
            print(f"  {key} -> {value}")
            
        # Check for common variable names
        expected_keys = [
            "Heart Rate", 
            "Beat-to-beat Interval", 
            "Steps", 
            "Stress Level"
        ]
        
        missing_keys = [key for key in expected_keys if key not in var_dict]
        if missing_keys:
            print(f"WARNING: var_dict is missing expected keys: {missing_keys}")
            
        return True
    except Exception as e:
        print(f"ERROR validating var_dict: {e}")
        return False

# Define helper function that fetches the correct point in time to plot the event/intervention
def get_time(row, instance_start, instance_end, start_time):
    if row["start_time"] + LOCAL_OFFSET < instance_start:
        return max(row["start_time"] + LOCAL_OFFSET, start_time)
    else:
        return max(row["start_time"] + LOCAL_OFFSET, instance_end)

def get_instances(user:str, instance_type:str, instance:str, mins_before:int, mins_after:int, var:str, var_dict:dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function that selects multiple instances of the same intervention, returns a dataframe of all instances.
    """
    # Convert minutes to ms
    ms_before, ms_after = mins_before * 60 * 1000, mins_after * 60 * 1000

    # Connect to the RDS database
    try:
        conn = get_rds_connection()
        cursor = conn.cursor()
    except Exception as e:
        empty_df = pd.DataFrame()
        empty_df.attrs['error_message'] = f"Database connection error: {str(e)}"
        return empty_df, None

    try:
        # Query the interventions table to get times for the given user and intervention
        cursor.execute(f"""
            SELECT start_time, end_time
            FROM {instance_type}
            WHERE name = %s
            AND {instance_type} = %s
        """, (user, instance))
        time_tuples = cursor.fetchall()  # List of tuples [(start_time_1,end_time_1), ...] (in ms)

        # If no results, try with the column name 'intervention' / 'event'
        if not time_tuples:
            cursor.execute(f"""
                SELECT start_time, end_time
                FROM {instance_type}
                WHERE name = %s
                AND {instance_type[:-1]} = %s
            """, (user, instance))
            time_tuples = cursor.fetchall()

        if not time_tuples:
            empty_df = pd.DataFrame()
            empty_df.attrs['error_message'] = f"No data available for {instance}. Please remember to tag the events/interventions you want to compare."
            return empty_df, None

        # Initialize dataframe of instances
        instances_df = pd.DataFrame(columns=["intervention_name", "event_name", "calendar_name"])

        for i, (init_instance_start, init_instance_end) in enumerate(time_tuples):
            instance_start = init_instance_start + LOCAL_OFFSET
            instance_end = init_instance_end + LOCAL_OFFSET
            start_time, end_time = instance_start - ms_before, instance_end + ms_after

            try:
                var_query = f"""
                    SELECT {var_dict[var]}, unix_timestamp_cleaned, timestamp_cleaned
                    FROM {user}_{var}
                    WHERE unix_timestamp_cleaned BETWEEN {start_time} AND {end_time}
                """

                # Create a dataframe with the selected data
                var_df = get_df_from_query(var_query, conn)
                
                if var_df.empty:
                    continue
                    
                var_df["timestamp_cleaned"] = pd.to_datetime(var_df["timestamp_cleaned"])

                # Ensure dtypes are correct
                var_df["unix_timestamp_cleaned"] = var_df["unix_timestamp_cleaned"].astype("int64")
                var_df[var] = var_df[var_dict[var]].astype("float64")
                var_df = var_df.set_index("unix_timestamp_cleaned")

                # Get interventions
                if instance_type != "interventions":
                    try:
                        intervention_query = f"""
                            SELECT start_time, end_time, interventions
                            FROM interventions
                            WHERE name = "{user}" 
                            AND (start_time BETWEEN {start_time-LOCAL_OFFSET} AND {end_time-LOCAL_OFFSET}
                            OR end_time BETWEEN {start_time-LOCAL_OFFSET} AND {end_time-LOCAL_OFFSET}
                            OR (start_time <= {start_time-LOCAL_OFFSET} AND end_time >= {end_time-LOCAL_OFFSET}))
                        """
                        intervention_df = get_df_from_query(intervention_query, conn)
                        if intervention_df.shape[0] > 0:
                            intervention_df["unix_timestamp_cleaned"] = intervention_df.apply(get_time, axis=1, args=(instance_start, instance_end, start_time))
                            intervention_df = intervention_df.set_index("unix_timestamp_cleaned")
                            intervention_df = intervention_df.rename({"interventions":"intervention_name", 
                                                                    "start_time":"intervention_start", 
                                                                    "end_time":"intervention_end"}, axis=1)
                            var_df = var_df.merge(intervention_df[["intervention_name","intervention_start","intervention_end"]], 
                                                left_index=True, right_index=True, how="outer")
                    except Exception:
                        # Continue without intervention data
                        pass

                # Get events
                if instance_type != "events":
                    try:
                        event_query = f"""
                            SELECT start_time, end_time, events
                            FROM events
                            WHERE name = "{user}" 
                            AND (start_time BETWEEN {start_time-LOCAL_OFFSET} AND {end_time-LOCAL_OFFSET}
                            OR end_time BETWEEN {start_time-LOCAL_OFFSET} AND {end_time-LOCAL_OFFSET}
                            OR (start_time <= {start_time-LOCAL_OFFSET} AND end_time >= {end_time-LOCAL_OFFSET}))
                        """
                        event_df = get_df_from_query(event_query, conn)
                        if event_df.shape[0] > 0:
                            event_df["unix_timestamp_cleaned"] = event_df.apply(get_time, axis=1, args=(instance_start, instance_end, start_time))
                            event_df = event_df.set_index("unix_timestamp_cleaned")
                            event_df = event_df.rename({"events":"event_name", 
                                                        "start_time":"event_start", 
                                                        "end_time":"event_end"}, axis=1)
                            var_df = var_df.merge(event_df[["event_name","event_start","event_end"]], 
                                                left_index=True, right_index=True, how="outer")
                    except Exception:
                        # Continue without events data
                        pass

                # Get calendar events
                if instance_type != "calendar events":
                    try:
                        start_time_dt = pd.to_datetime(start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
                        end_time_dt = pd.to_datetime(end_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
                        calendar_query = f"""
                            SELECT 
                                start_time,
                                end_time,
                                summary
                            FROM {user}_calendar_events
                            WHERE start_time BETWEEN '{start_time_dt}' AND '{end_time_dt}'
                            OR end_time BETWEEN '{start_time_dt}' AND '{end_time_dt}'
                            OR (start_time <= '{start_time_dt}' AND end_time >= '{end_time_dt}')
                        """
                        calendar_df = get_df_from_query(calendar_query, conn)
                        if calendar_df.shape[0] > 0:
                            calendar_df["start_time"] = pd.to_datetime(calendar_df["start_time"]).astype('int64') // 10**6  # Convert to ms
                            calendar_df["end_time"] = pd.to_datetime(calendar_df["end_time"]).astype('int64') // 10**6  # Convert to ms
                            calendar_df["unix_timestamp_cleaned"] = calendar_df.apply(get_time, axis=1, args=(instance_start, instance_end, start_time))
                            calendar_df = calendar_df.set_index("unix_timestamp_cleaned")
                            calendar_df = calendar_df.rename({"summary":"calendar_name", 
                                                        "start_time":"calendar_start", 
                                                        "end_time":"calendar_end"}, axis=1)
                            var_df = var_df.merge(calendar_df[["calendar_name","calendar_start","calendar_end"]],
                                                left_index=True, right_index=True, how="outer")
                    except Exception:
                        # Continue without calendar data
                        pass

                # Fill in NA values if present
                for col in ["intervention", "event", "calendar"]:
                    if f"{col}_name" in var_df.columns:
                        var_df = var_df.sort_index()
                        mask = var_df[f"{col}_name"].notna()
                        var_df.loc[mask, var] = var_df[var].interpolate(method="index")[mask]

                var_df = var_df.reset_index()

                # Determine if row is before, during, or after instance
                var_df["status"] = "during"
                var_df.loc[var_df["unix_timestamp_cleaned"] < instance_start, "status"] = "before"
                var_df.loc[var_df["unix_timestamp_cleaned"] > instance_end, "status"] = "after"

                # Calculate timedelta before and after intervention
                var_df["timedelta"] = pd.to_timedelta(var_df["unix_timestamp_cleaned"] - start_time, unit="ms")

                # Include duration of intervention
                var_df["duration"] = pd.to_timedelta(instance_end - instance_start, unit="ms")

                # Fetch sleep data for previous night
                try:
                    date_str = pd.to_datetime(instance_start, unit="ms").strftime(r"%Y-%m-%d")
                    date_str = date_str[1:] if date_str[0] == "0" else date_str  # Strip leading 0
                    
                    cursor.execute(f"""
                        SELECT durationInMs, overallSleepScore, overallSleepQualifier, calendarDate
                        FROM {user}_sleep_summary
                        WHERE calendarDate = '{date_str}'
                    """)
                    sleep_result = cursor.fetchone()
                    
                    if sleep_result:
                        sleep_duration_ms, sleep_score, sleep_label, _ = sleep_result
                        var_df["sleep_hours"] = pd.to_timedelta(sleep_duration_ms, unit="ms").total_seconds() / 60 / 60
                        var_df["sleep_score"] = sleep_score
                        var_df["sleep_label"] = sleep_label.title().strip() + " Sleep"
                    else:
                        # Try alternative sleep table
                        cursor.execute(f"""
                            SELECT score
                            FROM {user}_sleep
                            WHERE DATE(start_time) = '{date_str}'
                            ORDER BY start_time DESC
                            LIMIT 1
                        """)
                        alt_sleep_result = cursor.fetchone()
                        
                        if alt_sleep_result:
                            sleep_score = alt_sleep_result[0]
                            var_df["sleep_score"] = sleep_score
                            
                            # Add sleep label based on score
                            if sleep_score >= 80:
                                var_df["sleep_label"] = "Good Sleep"
                            elif sleep_score >= 60:
                                var_df["sleep_label"] = "Fair Sleep"
                            else:
                                var_df["sleep_label"] = "Poor Sleep"
                        else:
                            var_df["sleep_hours"] = np.nan
                            var_df["sleep_score"] = np.nan
                            var_df["sleep_label"] = "No Sleep Data"
                except Exception:
                    var_df["sleep_hours"] = np.nan
                    var_df["sleep_score"] = np.nan
                    var_df["sleep_label"] = "No Sleep Data"

                # Concatenate with all other instances
                instance_id = pd.to_datetime(instance_start, unit="ms") #.strftime("%a %d %b %Y, %I:%M%p")
                var_df["instance"] = instance_id
                instances_df = pd.concat([instances_df, var_df], axis=0)
            
            except Exception as e:
                continue

        # If no instances with data were found, return empty DataFrame with error message
        if instances_df.empty:
            empty_df = pd.DataFrame()
            empty_df.attrs['error_message'] = f"Found interventions for {instance}, but no matching {var} data in the time windows."
            return empty_df, None

        # Calculate the time since start/end of longest intervention/event
        max_duration = instances_df["duration"].max()
        
        def update_timedelta(row):
            if row["duration"] < max_duration and row["status"] == "after":
                return row["timedelta"] + max_duration - row["duration"]
            else:
                return row["timedelta"]
        
        instances_df["mins"] = instances_df.apply(update_timedelta, axis=1).dt.total_seconds() / 60

        # Create aggregate dataframe with only valid data
        instances_df[var] = instances_df[var].astype("float64")
        grouped_df = instances_df[instances_df[var] > 0].groupby(["instance", "status"]).agg({var: ["mean", "std"]}).reset_index()
        agg_df = grouped_df.pivot(index="instance", columns="status", values=[(var, "mean"), (var, "std")])

        return instances_df, agg_df
    
    except Exception as e:
        empty_df = pd.DataFrame()
        empty_df.attrs['error_message'] = f"Error retrieving data: {str(e)}"
        return empty_df, None
    
    finally:
        conn.close()

def get_mean_and_variance(user, intervention, var, X, var_dict, use="before"):
    """    This function selects data from the RDS database based on the user's name and the intervention.
    Then, it queries for var_dict[var] values that occurred between start_time and X minutes before each start_time.
    It aggregates these values into a list and returns the mean and variance.

    Args:
        user (str): The user's name.
        intervention (str): The intervention to filter in the interventions table.
        var (str): The variable to query in the dynamically generated table.
        X (int): Number of minutes to look back from the start_time.
        var_dict (dict): Dictionary containing variable mappings.
        use (str): Specifies whether to look "before" or "after" the start_time.

    Returns:
        mean (float): The mean of the selected variables.
        variance (float): The variance of the selected variables.
    """
    # Convert X minutes to milliseconds
    X_ms = X * 60 * 1000  # X minutes in milliseconds

    # Connect to the RDS database
    conn = get_rds_connection()
    cursor = conn.cursor()

    # Query the interventions table to get all start_times for the given user and intervention
    cursor.execute("""
        SELECT start_time
        FROM interventions
        WHERE name = %s
        AND interventions = %s
    """, (user, intervention))

    start_times = cursor.fetchall()  # List of tuples [(start_time_1,), (start_time_2,), ...]

    if not start_times:
        conn.close()
        return None, None  # If no start_times were found, return None for both mean and variance

    selected_values = []
    sequences = []

    # Loop over each start_time
    for start_time_tuple in start_times:
        start_time = start_time_tuple[0]  # Extract from tuple (in ms)

        # Calculate the time X minutes before or after start_time in milliseconds
        if use == "before":
            time_before = start_time - X_ms
            arg_sql = (time_before, start_time)
        elif use == "after":
            time_after = start_time + X_ms
            arg_sql = (start_time, time_after)
        else:
            raise ValueError("Invalid value for 'use'. Expected 'before' or 'after'.")
        # arg_sql = (0, 1000000000000000000)

        # Query the user-specific table to get the variable value within the specified time range
        cursor.execute(f"""
            SELECT {var_dict[var]}
            FROM {user}_{var}
            WHERE unix_timestamp_cleaned BETWEEN %s AND %s
        """, arg_sql)

        # Fetch all matching rows
        rows = cursor.fetchall()

        # Append all values from the query to the selected_values list
        selected_values.extend([row[0] for row in rows])
        sequences.append([row[0] for row in rows])

    conn.close()  # Close the database connection

    # If there are no selected values, return None
    if not selected_values:
        return None, None

    # Calculate the mean and variance using numpy
    mean = np.mean(selected_values)
    variance = np.var(selected_values)

    return mean, variance, sequences 

def get_calendar_events(user_name, start_date=None, end_date=None):
    """Fetch calendar events for a user within a date range."""
    conn = get_rds_connection()
    cursor = conn.cursor()
    
    try:
        query = f"""
            SELECT 
                start_time,
                end_time,
                summary,
                location,
                description,
                event_uid
            FROM {user_name}_calendar_events
        """
        
        if start_date and end_date:
            query += f" WHERE start_time BETWEEN '{start_date}' AND '{end_date}'"
        
        query += " ORDER BY start_time"
        
        df = get_df_from_query(query, conn)
        return df
    finally:
        cursor.close()
        conn.close()
        
def get_admin_name():
    conn = get_rds_connection()
    try:
        with conn.cursor() as cursor:
            # Query to get the admin name
            query = "SELECT name FROM admin_name LIMIT 1;"
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return "No admin name found."
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        conn.close()

        
if __name__ == "__main__":
    var_dict = {
        "Stress Level" : "stress",
        "Heart Rate" : "daily_heart_rate",
    }
    df = fetch_data("Stress Level", var_dict, "zw")
    print(df)