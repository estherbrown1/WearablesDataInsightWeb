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
       host="researchwearables.ct0y6yck0s1w.us-east-2.rds.amazonaws.com",  # Your RDS endpoint
       user="wearables2",  # Your RDS username
       password="wearables2",  # Your RDS password
       database="researchwearables",  # Database name on RDS
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
    """
    Fetches distinct event/intervention names for a user from the
    dedicated 'events' or 'interventions' tables.
    Corrected to select the 'name' column and filter by 'username'.
    """
    conn = None  # Initialize conn
    try:
    conn = get_rds_connection()
    cursor = conn.cursor()

        # Validate var_name is either 'events' or 'interventions'
        if var_name not in ['events', 'interventions']:
            print(f"Debug: Invalid table name requested: '{var_name}'")
            return []

        # Check if the specified table exists
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
        """, (var_name,))

        if cursor.fetchone()[0] == 0:
            print(f"Debug: Table '{var_name}' does not exist.")
            return []  # Table doesn't exist

        # Check if the 'username' and 'name' columns exist in the table
        cursor.execute(f"SHOW COLUMNS FROM {var_name}")
        columns = [col[0] for col in cursor.fetchall()]
        if 'username' not in columns or 'name' not in columns:
             print(f"Debug: Table '{var_name}' missing required 'username' or 'name' column.")
             return [] # Table missing required columns

        # --- CORRECTED QUERY ---
        # Select the distinct 'name' values for the specified 'username'
        query = f"""
            SELECT DISTINCT name
            FROM {var_name}
            WHERE username = %s
        """
        cursor.execute(query, (user_name,))
        # --- END CORRECTED QUERY ---

        # Fetch results and filter out None or empty values
        options = [row[0] for row in cursor.fetchall() if row[0] is not None and str(row[0]).strip() != '']
        print(f"Debug: Found options for {var_name} for user '{user_name}': {options}")
    return options

    except Exception as e:
        # Log the error for debugging
        print(f"Error in fetch_past_options for table '{var_name}', user '{user_name}': {str(e)}")
        # Consider adding traceback here if needed:
        # import traceback
        # print(traceback.format_exc())
        return [] # Return empty list on error

    finally:
        if conn:
            conn.close()


def save_other_response(user_name, response, var_name="events"):
    """Modified to add missing columns if needed"""
    conn = get_rds_connection()
    cursor = conn.cursor()

    try:
        # Check if column exists in users table
        cursor.execute("DESCRIBE users")
        columns = [column[0] for column in cursor.fetchall()]
        
        # Add column if it doesn't exist
        if var_name not in columns:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {var_name} TEXT")
            conn.commit()
        
        # Now we can safely fetch past responses
    cursor.execute(f"SELECT {var_name} FROM users WHERE name = %s", (user_name,))
    past_responses = cursor.fetchall()

    # Convert the results into a flat list of strings
    options = []
    if past_responses and past_responses[0][0]:  # Check if past responses exist and are not None
        options = past_responses[0][0].split("|||")
    
    options.append(response)
    new_options = "|||".join(options)

        # Update the user's response
    cursor.execute(f"UPDATE users SET {var_name} = %s WHERE name = %s", (new_options, user_name))
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
    conn.close()

def record_event_in_database(user, start_time, end_time, event_name, var_name,
                             category=None, notes=None, impact_feedback=None, event_type=None):
    """
    Records an event or intervention in the database.

    Args:
        user (str): Username
        start_time (int): Start time in milliseconds
        end_time (int): End time in milliseconds
        event_name (str): Name of the event
        var_name (str): Table name ('events' or 'interventions')
        category (str): Category label ('Event' or 'Intervention')
        notes (str): Optional user notes
        impact_feedback (str): Optional user feedback
        event_type (str): Source of annotation (e.g., 'manual', 'calendar', etc.)
    """
    conn = get_rds_connection()
    cursor = conn.cursor()

    try:
        # Ensure table exists with the full schema
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {var_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                start_time BIGINT,
                end_time BIGINT,
                {var_name} TEXT,
                category VARCHAR(255),
                notes TEXT,
                impact_feedback TEXT,
                username VARCHAR(255),
                event_type VARCHAR(255)
            )
        """)
        conn.commit()

        # Insert record
        cursor.execute(f"""
            INSERT INTO {var_name} (
                name, start_time, end_time, {var_name},
                category, notes, impact_feedback, username, event_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            event_name, start_time, end_time, event_name,
            category, notes, impact_feedback, user, event_type
        ))
    conn.commit()

        print(f"✅ Successfully inserted: {event_name} ({start_time} → {end_time}) into {var_name}")
    except Exception as e:
        print(f"❌ Error recording {event_name}: {e}")
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

def get_instances(user: str, instance_type: str, instance: str, mins_before: int, mins_after: int, var: str, var_dict: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function that selects multiple instances of the same intervention (or event)
    and returns a dataframe of all instances along with an aggregated dataframe.
    
    Parameters:
        user (str): The username.
        instance_type (str): Either 'events' or 'interventions'.
        instance (str): The specific event/intervention title.
        mins_before (int): Minutes before the event/intervention to include.
        mins_after (int): Minutes after the event/intervention to include.
        var (str): The variable key (e.g., 'stress').
        var_dict (dict): Mapping from display names to database column names.
    
    Returns:
        tuple: (instances_df, aggregate_df). Even if only one instance is found, the data are returned.
    """
    # Convert minutes to milliseconds
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
        # --- UPDATED QUERY ---
        # Filter by username and by the event/intervention title stored in the 'name' column.
        cursor.execute(f"""
            SELECT start_time, end_time
            FROM {instance_type}
            WHERE username = %s
            AND name = %s
        """, (user, instance))
        time_tuples = cursor.fetchall()  # List of tuples [(start_time_1, end_time_1), ...] in ms

        # Fallback: try using the singular column name if needed.
        if not time_tuples:
            cursor.execute(f"""
                SELECT start_time, end_time
                FROM {instance_type}
                WHERE username = %s
                AND {instance_type[:-1]} = %s
            """, (user, instance))
            time_tuples = cursor.fetchall()

        if not time_tuples:
            empty_df = pd.DataFrame()
            empty_df.attrs['error_message'] = f"No data available for {instance}. Please tag the events/interventions you want to compare."
            return empty_df, None

        # Initialize a DataFrame to hold all instance data
        instances_df = pd.DataFrame(columns=["intervention_name", "event_name", "calendar_name"])

        for i, (init_instance_start, init_instance_end) in enumerate(time_tuples):
            # Adjust instance times using the local offset
            instance_start = init_instance_start + LOCAL_OFFSET
            instance_end = init_instance_end + LOCAL_OFFSET
            start_window = instance_start - ms_before
            end_window = instance_end + ms_after

            try:
                var_query = f"""
                    SELECT {var_dict[var]}, unix_timestamp_cleaned, timestamp_cleaned
                    FROM {user}_{var}
                    WHERE unix_timestamp_cleaned BETWEEN {start_window} AND {end_window}
                """
                # Create a DataFrame with the selected variable data
                var_df = get_df_from_query(var_query, conn)
                if var_df.empty:
                    continue

                var_df["timestamp_cleaned"] = pd.to_datetime(var_df["timestamp_cleaned"])
                var_df["unix_timestamp_cleaned"] = var_df["unix_timestamp_cleaned"].astype("int64")
                var_df[var] = var_df[var_dict[var]].astype("float64")
                var_df = var_df.set_index("unix_timestamp_cleaned")

                # Merge in interventions data if instance_type is not 'interventions'
                if instance_type != "interventions":
                    try:
                        intervention_query = f"""
                            SELECT start_time, end_time, interventions
                            FROM interventions
                            WHERE username = %s
                            AND (
                                start_time BETWEEN {start_window - LOCAL_OFFSET} AND {end_window - LOCAL_OFFSET}
                                OR end_time BETWEEN {start_window - LOCAL_OFFSET} AND {end_window - LOCAL_OFFSET}
                                OR (start_time <= {start_window - LOCAL_OFFSET} AND end_time >= {end_window - LOCAL_OFFSET})
                            )
                        """
                        cursor.execute(intervention_query, (user,))
                        intervention_df = get_df_from_query(intervention_query, conn)
                        if intervention_df.shape[0] > 0:
                            intervention_df["unix_timestamp_cleaned"] = intervention_df.apply(get_time, axis=1, args=(instance_start, instance_end, start_window))
                            intervention_df = intervention_df.set_index("unix_timestamp_cleaned")
                            intervention_df = intervention_df.rename({
                                "interventions": "intervention_name", 
                                "start_time": "intervention_start", 
                                "end_time": "intervention_end"
                            }, axis=1)
                            var_df = var_df.merge(intervention_df[["intervention_name", "intervention_start", "intervention_end"]], 
                                                   left_index=True, right_index=True, how="outer")
                    except Exception:
                        pass

                # Merge in events data if instance_type is not 'events'
                if instance_type != "events":
                    try:
                        event_query = f"""
                            SELECT start_time, end_time, events
                            FROM events
                            WHERE username = %s
                            AND (
                                start_time BETWEEN {start_window - LOCAL_OFFSET} AND {end_window - LOCAL_OFFSET}
                                OR end_time BETWEEN {start_window - LOCAL_OFFSET} AND {end_window - LOCAL_OFFSET}
                                OR (start_time <= {start_window - LOCAL_OFFSET} AND end_time >= {end_window - LOCAL_OFFSET})
                            )
                        """
                        cursor.execute(event_query, (user,))
                        event_df = get_df_from_query(event_query, conn)
                        if event_df.shape[0] > 0:
                            event_df["unix_timestamp_cleaned"] = event_df.apply(get_time, axis=1, args=(instance_start, instance_end, start_window))
                            event_df = event_df.set_index("unix_timestamp_cleaned")
                            event_df = event_df.rename({
                                "events": "event_name", 
                                "start_time": "event_start", 
                                "end_time": "event_end"
                            }, axis=1)
                            var_df = var_df.merge(event_df[["event_name", "event_start", "event_end"]], 
                                                   left_index=True, right_index=True, how="outer")
                    except Exception:
                        pass

                # Merge in calendar events data if available
                if instance_type != "calendar events":
                    try:
                        start_time_dt = pd.to_datetime(start_window, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
                        end_time_dt = pd.to_datetime(end_window, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
                        calendar_query = f"""
                    SELECT start_time, end_time, summary
                    FROM {user}_calendar_events
                            WHERE start_time BETWEEN '{start_time_dt}' AND '{end_time_dt}'
                            OR end_time BETWEEN '{start_time_dt}' AND '{end_time_dt}'
                            OR (start_time <= '{start_time_dt}' AND end_time >= '{end_time_dt}')
                        """
                        calendar_df = get_df_from_query(calendar_query, conn)
                        if calendar_df.shape[0] > 0:
                            calendar_df["start_time"] = pd.to_datetime(calendar_df["start_time"]).astype('int64') // 10**6
                            calendar_df["end_time"] = pd.to_datetime(calendar_df["end_time"]).astype('int64') // 10**6
                            calendar_df["unix_timestamp_cleaned"] = calendar_df.apply(get_time, axis=1, args=(instance_start, instance_end, start_window))
                            calendar_df = calendar_df.set_index("unix_timestamp_cleaned")
                            calendar_df = calendar_df.rename({
                                "summary": "calendar_name", 
                                "start_time": "calendar_start", 
                                "end_time": "calendar_end"
                            }, axis=1)
                            var_df = var_df.merge(calendar_df[["calendar_name", "calendar_start", "calendar_end"]],
                                                   left_index=True, right_index=True, how="outer")
                    except Exception:
                        pass

                # Interpolate missing values for any related event columns if present
                for col in ["intervention", "event", "calendar"]:
                    if f"{col}_name" in var_df.columns:
                        var_df = var_df.sort_index()
                        mask = var_df[f"{col}_name"].notna()
                        var_df.loc[mask, var] = var_df[var].interpolate(method="index")[mask]

                var_df = var_df.reset_index()

                # Determine if each row is before, during, or after the instance
                var_df["status"] = "during"
                var_df.loc[var_df["unix_timestamp_cleaned"] < instance_start, "status"] = "before"
                var_df.loc[var_df["unix_timestamp_cleaned"] > instance_end, "status"] = "after"

                # Calculate timedelta (in ms) from the start of the window
                var_df["timedelta"] = pd.to_timedelta(var_df["unix_timestamp_cleaned"] - start_window, unit="ms")

                # Include the duration of the instance
                var_df["duration"] = pd.to_timedelta(instance_end - instance_start, unit="ms")

                # Fetch sleep data
                try:
                    today_dt = pd.to_datetime(instance_start, unit="ms")
                    today_str = today_dt.strftime(r"%Y-%m-%d")
                    tmr_dt = today_dt + pd.Timedelta(days=1)
                    tmr_str = tmr_dt.strftime(r"%Y-%m-%d")
                    
                    sleep_query = f"""
                        SELECT durationInMs, overallSleepScore, overallSleepQualifier, calendarDate
                        FROM {user}_sleep_summary
                        WHERE calendarDate = '{today_str}' OR calendarDate = '{tmr_str}'
                    """
                    sleep_df = get_df_from_query(sleep_query, conn)
                    
                    if sleep_df.empty:
                        # Try alternative sleep table
                cursor.execute(f"""
                    SELECT score
                    FROM {user}_sleep
                            WHERE DATE(start_time) = '{today_str}'
                    ORDER BY start_time DESC
                    LIMIT 1
                        """)
                        alt_sleep_result = cursor.fetchone()
                        
                        if alt_sleep_result:
                            sleep_score = alt_sleep_result[0]
                            var_df["sleep_score"] = sleep_score
                    
                    # Add sleep label based on score
                            if sleep_score >= 90:
                                var_df["sleep_label"] = "Excellent"
                            elif sleep_score >= 80:
                                var_df["sleep_label"] = "Good"
                    elif sleep_score >= 60:
                                var_df["sleep_label"] = "Fair"
                            else:
                                var_df["sleep_label"] = "Poor"
                        else:
                            for day in ["today","tmr"]:
                                var_df[f"{day}_sleep_hours"] = None
                                var_df[f"{day}_sleep_score"] = None
                                var_df[f"{day}_sleep_label"] = "No Sleep Data"
                    else:
                        # Sleep data for last night
                        today_df = sleep_df[sleep_df["calendarDate"] == today_str]
                        tmr_df = sleep_df[sleep_df["calendarDate"] == tmr_str]
                        for df, day in zip([today_df, tmr_df],["today", "tmr"]):
                            if len(df) == 1:
                                sleep_dur, sleep_score, sleep_label = df.iloc[0][["durationInMs", "overallSleepScore", "overallSleepQualifier"]]
                else:
                                sleep_dur, sleep_score, sleep_label = None, None, None
                            var_df[f"{day}_sleep_hours"] = pd.to_timedelta(sleep_dur, unit="ms").total_seconds() / 60 / 60
                            var_df[f"{day}_sleep_score"] = sleep_score
                            var_df[f"{day}_sleep_label"] = f"{sleep_label.title().strip()} Sleep"
                        
                except Exception:
                    for day in ["today","tmr"]:
                        var_df[f"{day}_sleep_hours"] = None
                        var_df[f"{day}_sleep_score"] = None
                        var_df[f"{day}_sleep_label"] = "No Sleep Data"

                # Fetch steps data
                try:

                    yday_dt = today_dt - pd.Timedelta(days=1)
                    yday_str = yday_dt.strftime(r"%Y-%m-%d")
                    steps_query = f"""
                        SELECT steps
                        FROM {user}_daily_summary
                        where calendarDate = '{yday_str}'
                        LIMIT 1
                    """
                    yday_steps = get_df_from_query(steps_query, conn).iloc[0,0]
                    var_df["steps_yesterday"] = f"{yday_steps:,d} Steps"
                except Exception:
                    var_df["steps_yesterday"] = ""

                # Tag this instance using its start time as identifier
                instance_id = pd.to_datetime(instance_start, unit="ms")
                var_df["instance"] = instance_id

                # Concatenate this instance data with any previous instances
                instances_df = pd.concat([instances_df, var_df], axis=0)
            
            except Exception as e:
                # Skip this instance if an error occurs
                continue

        # If no instance data were found, return an empty DataFrame with an error message.
        if instances_df.empty:
            empty_df = pd.DataFrame()
            empty_df.attrs['error_message'] = f"Found instances for {instance}, but no matching {var} data in the time windows."
            return empty_df, None

        # Ensure that the expected variable column exists.
        if var not in instances_df.columns:
            # If missing, create it with NaN values.
            instances_df[var] = None

        # Calculate the adjusted time difference (in minutes) for plotting.
        max_duration = instances_df["duration"].max()
        def update_timedelta(row):
            if row["duration"] < max_duration and row["status"] == "after":
                return row["timedelta"] + max_duration - row["duration"]
            else:
                return row["timedelta"]
        instances_df["mins"] = instances_df.apply(update_timedelta, axis=1).dt.total_seconds() / 60

        # Create an aggregate DataFrame with only valid data.
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
        
def print_table_structure(table_name="events"):
    with get_rds_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"DESCRIBE {table_name}")
            structure = cursor.fetchall()
            header = ["Field", "Type", "Null", "Key", "Default", "Extra"]
            print(f"--- Structure of {table_name} ---")
            print(header)
            for row in structure:
                print(row)
            print("\n")

def print_annotations_for_username(username, table_name="events"):
    with get_rds_connection() as conn:
        with conn.cursor() as cursor:
            query = f"SELECT * FROM {table_name} WHERE username = %s"
            cursor.execute(query, (username,))
            rows = cursor.fetchall()
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
            else:
                columns = []
            print(f"--- Annotations for '{username}' ---")
            print("Columns:", columns)
            print("Total rows:", len(rows))
            for row in rows:
                print(row)
            print("\n")

        
# if __name__ == "__main__":
#     var_dict = {
#         "Stress Level" : "stress",
#         "Heart Rate" : "daily_heart_rate",
#     }
#     df = fetch_data("Stress Level", var_dict, "zw")
#     print(df)
if __name__ == "__main__":
    # Print the structure of the events table
    print_table_structure("events")
    
    # Compare the annotations for the two usernames:
    print_annotations_for_username("g_study_data", "events")
    print_annotations_for_username("e_study_c", "events")