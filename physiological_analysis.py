import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
from datetime import datetime, timedelta
import re
from sql_utils import get_rds_connection
import streamlit as st
import altair as alt  # Add this import
import numpy as np

# ML imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap


# Import VAR_DICT from your visualization module (adjust the path as needed)
from visualization_page import VAR_DICT  
from sql_utils import LOCAL_OFFSET

# ------------------ CLEAR OLD ROWS FUNCTION ------------------ #
def clear_previous_data(username: str):
    """
    Deletes all existing rows for the given username from
    the events and interventions tables.
    """
    with get_rds_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM events WHERE username = %s", (username,))
            cursor.execute("DELETE FROM interventions WHERE username = %s", (username,))
            conn.commit()
    st.warning(f"Cleared all old data for user '{username}' from events and interventions.")

# ------------------ HELPER FUNCTIONS ------------------ #
def get_table_data(username: str, table_name: str) -> pd.DataFrame:
    """
    Fetches data from the specified table (either 'events' or 'interventions')
    for a given username.
    """
    try:
        with get_rds_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM information_schema.tables
                    WHERE table_name = '{table_name}'
                """)
                if cursor.fetchone()[0] == 0:
                    st.warning(f"Table {table_name} does not exist")
                    return pd.DataFrame()
                
                cursor.execute(f"DESCRIBE {table_name}")
                columns = [row[0] for row in cursor.fetchall()]
                select_cols = [col for col in ["id", "username", "name", "start_time", "end_time", "category", "notes", "impact_feedback"] if col in columns]
                
                if not select_cols:
                    st.warning(f"No usable columns found in {table_name}")
                    return pd.DataFrame()
                
                query = f"SELECT {', '.join(select_cols)} FROM {table_name} WHERE username = %s"
                cursor.execute(query, (username,))
                rows = cursor.fetchall()
                if not rows:
                    st.warning(f"No data found in {table_name} for user {username}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(rows, columns=select_cols)
                for col in ["start_time", "end_time"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col] + LOCAL_OFFSET, unit='ms')
                df["name"] = df["name"].fillna("Unspecified") if "name" in df.columns else "Unspecified"
                df["instance_type"] = "Event" if table_name == "events" else "Intervention"
                df["category"] = "Event" if table_name == "events" else "Intervention"
                for col in ["notes", "impact_feedback"]:
                    if col not in df.columns:
                        df[col] = ""
                return df
    except Exception as e:
        st.error(f"Error retrieving data from {table_name}: {str(e)}")
        return pd.DataFrame()

def extract_sentiment(notes):
    """Extracts sentiment classification from notes text."""
    if not isinstance(notes, str):
        return 'Unknown'
    notes_lower = notes.lower()
    positive_patterns = [
        r'\b(positive|good|helpful|enjoy|calming|inspiring)\b',
        r'positive\s+in\s+the\s+long\s+term',
        r'should\s+be\s+positive',
        r'really\s+enjoyed'
    ]
    negative_patterns = [
        r'\b(negative|bad|anxious|worried|less engaged|dismissed)\b',
        r'less\s+engaged',
        r'kind of\s+less\s+engaged',
        r'left\s+me\s+less\s+engaged',
        r'made\s+me\s+slightly\s+anxious'
    ]
    neutral_patterns = [
        r'\bneutral',
        r'neutrally\s+engaging',
        r'neutral,\s+maybe\s+positive'
    ]
    for pattern in positive_patterns:
        if re.search(pattern, notes_lower):
            return 'Positive'
    for pattern in negative_patterns:
        if re.search(pattern, notes_lower):
            return 'Negative'
    for pattern in neutral_patterns:
        if re.search(pattern, notes_lower):
            return 'Neutral'
    return 'Unknown'

def extract_reported_impact(feedback):
    """Extracts a classification of reported impact from the impact feedback text."""
    if not isinstance(feedback, str) or feedback.strip() == "":
        return 'Unknown'
    feedback_lower = feedback.lower()
    if "positive" in feedback_lower:
        return "Positive"
    elif "negative" in feedback_lower:
        return "Negative"
    elif "neutral" in feedback_lower:
        return "Neutral"
    else:
        return "Unknown"

def analyze_metric_change(username: str, metric_name: str, pre_start, pre_end, post_start, post_end):
    """
    Analyzes changes in a physiological metric before and after an event.
    Returns percent change, pre_avg, post_avg, and a flag for data availability.
    """
    # NOTE: Verbose debug output is commented out but we can use it to debug the analysis
    # If you need to see the detailed analysis info again, uncomment the st.write statements
    try:
        # st.write(f"Analyzing {metric_name} from {pre_start.strftime('%Y-%m-%d %H:%M:%S')} to {post_end.strftime('%Y-%m-%d %H:%M:%S')}")
        pre_start_ms = int(pre_start.timestamp() * 1000)
        pre_end_ms = int(pre_end.timestamp() * 1000) 
        post_start_ms = int(post_start.timestamp() * 1000)
        post_end_ms = int(post_end.timestamp() * 1000)
        with get_rds_connection() as conn:
            with conn.cursor() as cursor:
                table_name = f"{username}_{metric_name}"
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_name = %s
                """, (table_name,))
                if cursor.fetchone()[0] == 0:
                    # st.warning(f"Table {table_name} not found.")
                    return {'percent_change': 0.0, 'pre_avg': None, 'post_avg': None, 'data_available': False}
                cursor.execute(f"SHOW COLUMNS FROM {table_name}")
                columns = [col[0] for col in cursor.fetchall()]
                value_column = None
                try:
                    var_name = VAR_DICT.get(metric_name, metric_name)
                    if var_name in columns:
                        value_column = var_name
                except (ImportError, KeyError):
                    pass
                if not value_column:
                    if metric_name == 'stress' and 'stressLevel' in columns:
                        value_column = 'stressLevel'
                    elif metric_name in ['bbi', 'rmssd', 'steps'] and metric_name in columns:
                        value_column = metric_name
                    elif metric_name == 'steps' and 'totalSteps' in columns:
                        value_column = 'totalSteps'
                if not value_column:
                    # st.warning(f"No suitable value column found in {table_name}.")
                    return {'percent_change': 0.0, 'pre_avg': None, 'post_avg': None, 'data_available': False}
                
                if 'timestamp_cleaned' in columns:
                    cursor.execute(f"""
                        SELECT MIN(unix_timestamp_cleaned), MAX(unix_timestamp_cleaned)
                        FROM {table_name}
                        WHERE unix_timestamp_cleaned > 0
                    """)
                    min_ms, max_ms = cursor.fetchone()
                    from datetime import datetime
                    try:
                        if min_ms is not None and max_ms is not None:
                            min_ms = float(min_ms) if isinstance(min_ms, str) else min_ms
                            max_ms = float(max_ms) if isinstance(max_ms, str) else max_ms
                            # st.write("Timezone verification:")
                            # st.write(f"  Unix range: {min_ms} to {max_ms}")
                            # st.write(f"  Datetime: {datetime.fromtimestamp(min_ms/1000)} to {datetime.fromtimestamp(max_ms/1000)}")
                    except Exception as e:
                        # st.warning(f"Error converting timestamps: {e}")
                        pass
                
                cursor.execute(f"""
                    SELECT COUNT(*), AVG({value_column})
                    FROM {table_name}
                    WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                """, (pre_start_ms, pre_end_ms))
                pre_count, pre_avg = cursor.fetchone()
                cursor.execute(f"""
                    SELECT COUNT(*), AVG({value_column})
                    FROM {table_name}
                    WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                """, (post_start_ms, post_end_ms))
                post_count, post_avg = cursor.fetchone()
                
                # DEBUGGGING LOGS (already commented out)
                
                # if pre_count > 0:
                #     cursor.execute(f"""
                #         SELECT unix_timestamp_cleaned, {value_column}
                #         {', timestamp_cleaned' if 'timestamp_cleaned' in columns else ''}
                #         FROM {table_name}
                #         WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                #         ORDER BY unix_timestamp_cleaned LIMIT 3
                #     """, (pre_start_ms, pre_end_ms))
                #     pre_samples = cursor.fetchall()
                #     st.write(f"Pre-window sample ({pre_count} records):")
                #     for sample in pre_samples:
                #         st.write(f"  Unix: {sample[0]}, Value: {sample[1]}")
                # if post_count > 0:
                #     cursor.execute(f"""
                #         SELECT unix_timestamp_cleaned, {value_column}
                #         {', timestamp_cleaned' if 'timestamp_cleaned' in columns else ''}
                #         FROM {table_name}
                #         WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                #         ORDER BY unix_timestamp_cleaned LIMIT 3
                #     """, (post_start_ms, post_end_ms))
                #     post_samples = cursor.fetchall()
                #     st.write(f"Post-window sample ({post_count} records):")
                #     for sample in post_samples:
                #         st.write(f"  Unix: {sample[0]}, Value: {sample[1]}")
                
                percent_change = 0.0
                if pre_avg is not None and post_avg is not None and pre_avg != 0 and pre_count > 0 and post_count > 0:
                    pre_avg_float = float(pre_avg)
                    post_avg_float = float(post_avg)
                    percent_change = ((post_avg_float - pre_avg_float) / pre_avg_float) * 100
                    if metric_name in ["stress", "daily_heart_rate", "respiration"]:
                        direction = "decreased" if percent_change < 0 else "increased"
                        effect = "Positive" if percent_change < 0 else "Negative"
                    elif metric_name in ["bbi", "rmssd", "hrv"]:
                        direction = "increased" if percent_change > 0 else "decreased"
                        effect = "Positive" if percent_change > 0 else "Negative"
                    else:
                        direction = "changed"
                        effect = "Neutral"
                    # st.write(f"{metric_name} {direction} by {abs(percent_change):.2f}% ({effect} effect)")
                    return {
                        'percent_change': float(percent_change),
                        'pre_avg': float(pre_avg_float),
                        'post_avg': float(post_avg_float),
                        'data_available': True,
                        'is_positive': effect == "Positive"
                    }
                else:
                    # st.warning(f"Insufficient data for {metric_name} change. Pre-count: {pre_count}, Post-count: {post_count}")
                    return {
                        'percent_change': 0.0,
                        'pre_avg': float(pre_avg) if pre_avg is not None else None,
                        'post_avg': float(post_avg) if post_avg is not None else None,
                        'data_available': False
                    }
    
    except Exception as e:
        # st.error(f"Error analyzing {metric_name} change: {str(e)}")
        # import traceback
        # st.error(traceback.format_exc())
        return {'percent_change': 0.0, 'pre_avg': None, 'post_avg': None, 'data_available': False}

# ------------------ NEW COMPOSITE IMPACT SCORE FUNCTION ------------------ #
def calculate_impact_score_all_features(df: pd.DataFrame) -> pd.Series:
    """
    Automatically calculates a composite impact score using all feature columns
    that represent changes (i.e. columns ending with '_change'). Each metric is 
    standardized (z-score) and sign adjusted such that higher values always indicate 
    a more positive effect.

    For metrics where a decrease is beneficial (e.g., stress or heart rate), the 
    standardized value is multiplied by -1.

    The composite impact score is the average of these standardized scores.

    This means:
    1. For each metric (like stress_change, rmssd_change, bbi_change, etc.), compute the z-score:
         (value - mean) / standard deviation.
    2. For metrics where a lower raw value is better (e.g., stress_change), multiply the z-score by -1.
    3. Average the resulting standardized values.
    """
    # Identify all columns that represent a change
    metric_cols = [col for col in df.columns if col.endswith('_change')]
    if not metric_cols:
        return pd.Series(0.0, index=df.index)
    
    # Define which metrics benefit from a decrease.
    decrease_beneficial = ['stress_change', 'heart_rate_change']
    
    standardized_metrics = pd.DataFrame(index=df.index)
    
    for col in metric_cols:
        # Determine sign factor: if a decrease is good, flip sign.
        sign_factor = -1 if any(key in col for key in decrease_beneficial) else 1
        
        # Calculate z-score for the column
        col_mean = df[col].mean()
        col_std = df[col].std() if df[col].std() != 0 else 1e-6  # avoid division by zero
        standardized_metrics[col] = sign_factor * ((df[col] - col_mean) / col_std)
    
    # The composite impact score is the average of all standardized metrics
    composite_score = standardized_metrics.mean(axis=1)
    return composite_score

# ------------------ MAIN ANALYSIS FUNCTION ------------------ #
def analyze_physiological_impact(username: str, hours_window: int = 2) -> pd.DataFrame:
    """
    Analyzes physiological patterns around events and interventions.
    
    For each event/intervention, it computes pre-event and post-event averages,
    calculates percent changes, extracts sentiment and user feedback, and
    computes a composite impact score.
    
    NOTE: For HRV, use 'rmssd' as defined in VAR_DICT.
    """
    st.info("Analyzing physiological patterns around events and interventions...")
    events_df = get_table_data(username, 'events')
    interventions_df = get_table_data(username, 'interventions')
    st.write(f"Found {len(events_df)} events and {len(interventions_df)} interventions")
    if not events_df.empty:
        st.dataframe(events_df[['name', 'instance_type', 'start_time', 'end_time']].head())
    if not interventions_df.empty:
        st.dataframe(interventions_df[['name', 'instance_type', 'start_time', 'end_time']].head())
    if events_df.empty and interventions_df.empty:
        st.warning(f"No events or interventions found for user {username}")
        return pd.DataFrame()
    all_instances = pd.concat([events_df, interventions_df], ignore_index=True)
    # Compute day of week and time of day from the start time
    all_instances['time_of_day'] = all_instances['start_time'].apply(
        lambda x: 'Morning' if 5 <= x.hour < 12 else 'Afternoon' if 12 <= x.hour < 17 else 'Evening' if 17 <= x.hour < 21 else 'Night'
    )
    all_instances['day_of_week'] = all_instances['start_time'].dt.day_name()
    st.dataframe(all_instances[['name', 'instance_type', 'start_time', 'end_time']].head(3))
    all_instances['sentiment'] = all_instances['notes'].apply(extract_sentiment)
    all_instances['reported_impact'] = all_instances['impact_feedback'].apply(extract_reported_impact)
    results = []
    for _, instance in all_instances.iterrows():
        start_time = instance['start_time']
        end_time = instance['end_time']
        duration_mins = (end_time - start_time).total_seconds() / 60
        pre_window = (start_time - pd.Timedelta(hours=hours_window), start_time)
        post_window = (end_time, end_time + pd.Timedelta(hours=hours_window))
        metrics_analysis = {}
        pre_metrics = {}
        post_metrics = {}
        for metric in ['rmssd', 'bbi', 'steps', 'stress']:
            metric_change = analyze_metric_change(username, metric, pre_window[0], pre_window[1], post_window[0], post_window[1])
            metrics_analysis[f'{metric}_change'] = metric_change['percent_change']
            metrics_analysis[f'{metric}_pre_avg'] = metric_change['pre_avg']
            metrics_analysis[f'{metric}_post_avg'] = metric_change['post_avg']
            pre_metrics[metric] = metric_change['pre_avg']
            post_metrics[metric] = metric_change['post_avg']
        stress_reduction = False
        if pre_metrics['stress'] is not None and post_metrics['stress'] is not None:
            stress_reduction = pre_metrics['stress'] > post_metrics['stress']
        pre_stress_state = 'Unknown'
        if pre_metrics['stress'] is not None:
            if pre_metrics['stress'] < 30:
                pre_stress_state = 'Low'
            elif pre_metrics['stress'] < 60:
                pre_stress_state = 'Medium'
            else:
                pre_stress_state = 'High'
        row_dict = {
            'username': instance.get('username', username),
            'instance_type': instance.get('instance_type', 'Event'),
            'name': instance.get('name', 'Unspecified'),
            'start_time': start_time,
            'end_time': end_time,
            'duration_minutes': duration_mins,
            'sentiment': instance['sentiment'],
            'reported_impact': instance.get('reported_impact', 'Unknown'),
            'notes': instance.get('notes', ''),
            'time_of_day': instance['time_of_day'],
            'day_of_week': instance['day_of_week'],
            'pre_stress_state': pre_stress_state,
            'stress_reduction': stress_reduction,
            **metrics_analysis
        }
        results.append(row_dict)
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['impact_score'] = calculate_impact_score_all_features(results_df)
        st.dataframe(results_df[['name', 'instance_type', 'impact_score']].head(3))
        st.session_state['time_patterns'] = results_df.groupby(['instance_type', 'name', 'time_of_day']).agg({
            'stress_reduction': lambda x: (x == True).mean(),
            'impact_score': 'mean'
        }).reset_index()
        st.session_state['day_patterns'] = results_df.groupby(['instance_type', 'name', 'day_of_week']).agg({
            'stress_reduction': lambda x: (x == True).mean(),
            'impact_score': 'mean'
        }).reset_index()
        st.session_state['stress_patterns'] = results_df.groupby(['instance_type', 'name', 'pre_stress_state']).agg({
            'stress_reduction': lambda x: (x == True).mean(),
            'impact_score': 'mean'
        }).reset_index()
        results_df['duration_category'] = pd.cut(
            results_df['duration_minutes'],
            bins=[0, 15, 30, 60, float('inf')],
            labels=['<15 min', '15-30 min', '30-60 min', '>60 min']
        )
        st.session_state['duration_patterns'] = results_df.groupby(['instance_type', 'name', 'duration_category']).agg({
            'stress_reduction': lambda x: (x == True).mean(),
            'impact_score': 'mean'
        }).reset_index()
    return results_df


def calculate_bbi_continuous_change(username: str, intervention_name: str, minutes_range=60) -> pd.DataFrame:
    """
    Calculate continuous BBI changes for a specified time after an intervention.
    Returns a DataFrame with continuous timestamps and corresponding BBI changes.
    Uses a rolling baseline approach where each point is compared to the previous 5-minute window.
    """
    with get_rds_connection() as conn:
        with conn.cursor() as cursor:
            # First check if the BBI table exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'researchwearables' 
                AND table_name = %s
            """, (f'{username}_bbi',))
            
            if cursor.fetchone()[0] == 0:
                print(f"Warning: BBI table '{username}_bbi' does not exist in the database.")
                return pd.DataFrame()
            
            # Check if the table has any data
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM {username}_bbi
            """)
            if cursor.fetchone()[0] == 0:
                print(f"Warning: BBI table '{username}_bbi' exists but contains no data.")
                return pd.DataFrame()
            
            # Get the intervention's start and end times
            cursor.execute("""
                SELECT id, start_time, end_time 
                FROM interventions 
                WHERE username = %s AND name = %s
            """, (username, intervention_name))
            intervention_times = cursor.fetchall()
            
            if not intervention_times:
                print(f"Warning: No intervention times found for {intervention_name}")
                return pd.DataFrame()
            
            all_results = []
            for intervention_id, start_time, end_time in intervention_times:
                # Convert to numeric if they're strings
                if isinstance(start_time, str):
                    start_time = float(start_time)
                if isinstance(end_time, str):
                    end_time = float(end_time)
                
                # Get continuous BBI data from well before the intervention to the end of our analysis window
                # We need to get data from before the intervention to calculate rolling baselines
                pre_start = start_time - (15 * 60 * 1000)  # 15 minutes before intervention start to have enough context
                post_end_time = end_time + (minutes_range * 60 * 1000)  # Our full analysis window after intervention
                
                cursor.execute(f"""
                    SELECT unix_timestamp_cleaned, bbi
                    FROM {username}_bbi
                    WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                    ORDER BY unix_timestamp_cleaned
                """, (pre_start, post_end_time))
                
                continuous_data = cursor.fetchall()
                
                if not continuous_data:
                    print(f"Warning: No BBI data found for {intervention_name} (ID: {intervention_id})")
                    continue
                
                # Convert to DataFrame for easier processing
                df = pd.DataFrame(continuous_data, columns=['unix_timestamp', 'bbi'])
                
                # Ensure numeric data types
                df['unix_timestamp'] = pd.to_numeric(df['unix_timestamp'])
                df['bbi'] = pd.to_numeric(df['bbi'])
                
                # Calculate minutes relative to intervention end
                df['minutes_rel_end'] = (df['unix_timestamp'] - end_time) / (60 * 1000)
                
                # Filter to only post-intervention data for results
                post_df = df[df['unix_timestamp'] >= end_time].copy()
                
                if post_df.empty:
                    print(f"Warning: No post-intervention BBI data for {intervention_name} (ID: {intervention_id})")
                    continue
                
                # For each post-intervention point, calculate a rolling baseline
                for idx, row in post_df.iterrows():
                    current_time = row['unix_timestamp']
                    rolling_baseline_start = current_time - (5 * 60 * 1000)  # 5 minutes before current point
                    
                    # Get the rolling baseline window data
                    baseline_data = df[(df['unix_timestamp'] >= rolling_baseline_start) & 
                                      (df['unix_timestamp'] < current_time)]
                    
                    if not baseline_data.empty:
                        rolling_baseline_avg = baseline_data['bbi'].mean()
                        
                        # Calculate absolute change from rolling baseline
                        absolute_change = row['bbi'] - rolling_baseline_avg
                        
                        all_results.append({
                            'unix_timestamp': current_time,
                            'minutes_after': row['minutes_rel_end'],
                            'bbi_change': absolute_change,  # Absolute change in ms
                            'intervention': intervention_name,
                            'intervention_id': intervention_id
                        })
            
            result_df = pd.DataFrame(all_results)
            
            # Apply smoothing to the resulting data if needed
            if not result_df.empty:
                # Sort by minutes_after to ensure proper smoothing
                result_df = result_df.sort_values('minutes_after')
                
                # Apply a rolling window smoothing to reduce noise
                for (intervention, intervention_id), group_df in result_df.groupby(['intervention', 'intervention_id']):
                    mask = ((result_df['intervention'] == intervention) & 
                           (result_df['intervention_id'] == intervention_id))
                    
                    # Apply smoothing with a 5-point window
                    result_df.loc[mask, 'bbi_change'] = group_df['bbi_change'].rolling(
                        window=5, center=True, min_periods=1
                    ).mean()
            
            return result_df

def calculate_hrv_continuous_change(username: str, intervention_name: str, minutes_range=60) -> pd.DataFrame:
    """
    Calculate continuous HRV changes for a specified time after an intervention.
    Returns a DataFrame with continuous timestamps and corresponding HRV changes.
    Uses a rolling baseline approach where each point is compared to the previous 5-minute window.
    """
    with get_rds_connection() as conn:
        with conn.cursor() as cursor:
            # First check if the RMSSD table exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'researchwearables' 
                AND table_name = %s
            """, (f'{username}_rmssd',))
            
            if cursor.fetchone()[0] == 0:
                print(f"Warning: RMSSD table '{username}_rmssd' does not exist in the database.")
                return pd.DataFrame()
            
            # Check if the table has any data
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM {username}_rmssd
            """)
            if cursor.fetchone()[0] == 0:
                print(f"Warning: RMSSD table '{username}_rmssd' exists but contains no data.")
                return pd.DataFrame()
            
            # Get the intervention's start and end times
            cursor.execute("""
                SELECT id, start_time, end_time 
                FROM interventions 
                WHERE username = %s AND name = %s
            """, (username, intervention_name))
            intervention_times = cursor.fetchall()
            
            if not intervention_times:
                print(f"Warning: No intervention times found for {intervention_name}")
                return pd.DataFrame()
            
            all_results = []
            for intervention_id, start_time, end_time in intervention_times:
                # Convert to numeric if they're strings
                if isinstance(start_time, str):
                    start_time = float(start_time)
                if isinstance(end_time, str):
                    end_time = float(end_time)
                
                # Get continuous RMSSD data from well before the intervention to the end of our analysis window
                # We need to get data from before the intervention to calculate rolling baselines
                pre_start = start_time - (15 * 60 * 1000)  # 15 minutes before intervention start to have enough context
                post_end_time = end_time + (minutes_range * 60 * 1000)  # Our full analysis window after intervention
                
                cursor.execute(f"""
                    SELECT unix_timestamp_cleaned, rmssd
                    FROM {username}_rmssd
                    WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                    ORDER BY unix_timestamp_cleaned
                """, (pre_start, post_end_time))
                
                continuous_data = cursor.fetchall()
                
                if not continuous_data:
                    print(f"Warning: No RMSSD data found for {intervention_name} (ID: {intervention_id})")
                    continue
                
                # Convert to DataFrame for easier processing
                df = pd.DataFrame(continuous_data, columns=['unix_timestamp', 'rmssd'])
                
                # Ensure numeric data types
                df['unix_timestamp'] = pd.to_numeric(df['unix_timestamp'])
                df['rmssd'] = pd.to_numeric(df['rmssd'])
                
                # Calculate minutes relative to intervention end
                df['minutes_rel_end'] = (df['unix_timestamp'] - end_time) / (60 * 1000)
                
                # Filter to only post-intervention data for results
                post_df = df[df['unix_timestamp'] >= end_time].copy()
                
                if post_df.empty:
                    print(f"Warning: No post-intervention RMSSD data for {intervention_name} (ID: {intervention_id})")
                    continue
                
                # For each post-intervention point, calculate a rolling baseline
                for idx, row in post_df.iterrows():
                    current_time = row['unix_timestamp']
                    rolling_baseline_start = current_time - (5 * 60 * 1000)  # 5 minutes before current point
                    
                    # Get the rolling baseline window data
                    baseline_data = df[(df['unix_timestamp'] >= rolling_baseline_start) & 
                                      (df['unix_timestamp'] < current_time)]
                    
                    if not baseline_data.empty:
                        rolling_baseline_avg = baseline_data['rmssd'].mean()
                        
                        # Calculate absolute change from rolling baseline
                        absolute_change = row['rmssd'] - rolling_baseline_avg
                        
                        all_results.append({
                            'unix_timestamp': current_time,
                            'minutes_after': row['minutes_rel_end'],
                            'rmssd_change': absolute_change,  # Absolute change in ms
                            'intervention': intervention_name,
                            'intervention_id': intervention_id
                        })
            
            result_df = pd.DataFrame(all_results)
            
            # Apply smoothing to the resulting data if needed
            if not result_df.empty:
                # Sort by minutes_after to ensure proper smoothing
                result_df = result_df.sort_values('minutes_after')
                
                # Apply a rolling window smoothing to reduce noise
                for (intervention, intervention_id), group_df in result_df.groupby(['intervention', 'intervention_id']):
                    mask = ((result_df['intervention'] == intervention) & 
                           (result_df['intervention_id'] == intervention_id))
                    
                    # Apply smoothing with a 5-point window
                    result_df.loc[mask, 'rmssd_change'] = group_df['rmssd_change'].rolling(
                        window=5, center=True, min_periods=1
                    ).mean()
            
            return result_df

def visualize_analysis(results_df: pd.DataFrame, username: str):
    if results_df.empty:
        st.warning("No data available for visualization")
        return
    plot_df = results_df.copy()
    plot_df = plot_df[plot_df['name'] != 'Unspecified']
    if plot_df.empty:
        st.warning("No named events or interventions found for visualization.")
        st.info("Your data was imported, but all entries have 'Unspecified' as the name. Please import data with specific names or recategorize your entries.")
        return
    st.subheader(f"Physiological Impact Analysis for User: {username}")
    
    # Standardized Metrics Visualizations

    # BBI Change Plot (unchanged)
    st.write("### BBI Change by Event/Intervention Name")
    name_bbi_change = plot_df.groupby('name')[['bbi_pre_avg', 'bbi_post_avg']].mean().reset_index()
    name_bbi_change['bbi_pct_change'] = (name_bbi_change['bbi_post_avg'] - name_bbi_change['bbi_pre_avg']) / name_bbi_change['bbi_pre_avg'] * 100
    fig4, ax5 = plt.subplots(figsize=(12, 6))
    name_bbi_change = name_bbi_change.sort_values('bbi_pct_change', ascending=False)
    bbi_colors = ['green' if val > 0 else 'red' for val in name_bbi_change['bbi_pct_change']]
    ax5.bar(name_bbi_change['name'], name_bbi_change['bbi_pct_change'], color=bbi_colors)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.set_title("Average BBI Percent Change by Event/Intervention Name")
    ax5.set_xlabel("Event/Intervention")
    ax5.set_ylabel("Percent Change in BBI (%; higher is better)")
    ax5.set_xticklabels(name_bbi_change['name'], rotation=45, ha='right')
    ax5.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)

    # HRV Change Plot (Standard Resolution)
    st.write("### HRV Change by Event/Intervention Name (Standard Resolution)")
    name_rmssd_change = plot_df.groupby('name')[['rmssd_pre_avg', 'rmssd_post_avg']].mean().reset_index()
    name_rmssd_change['rmssd_pct_change'] = (name_rmssd_change['rmssd_post_avg'] - name_rmssd_change['rmssd_pre_avg']) / name_rmssd_change['rmssd_pre_avg'] * 100
    fig3, ax4 = plt.subplots(figsize=(12, 6))
    name_rmssd_change = name_rmssd_change.sort_values('rmssd_pct_change', ascending=False)
    rmssd_colors = ['green' if val > 0 else 'red' for val in name_rmssd_change['rmssd_pct_change']]
    ax4.bar(name_rmssd_change['name'], name_rmssd_change['rmssd_pct_change'], color=rmssd_colors)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_title("Average HRV Percent Change by Event/Intervention Name (Standard Resolution)")
    ax4.set_xlabel("Event/Intervention")
    ax4.set_ylabel("Percent Change in HRV (%; higher is better)")
    ax4.set_xticklabels(name_rmssd_change['name'], rotation=45, ha='right')
    ax4.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Calculate non-standardized % change for stress, first averaging pre-post effects then taking the percent change
    st.write("### Which events/interventions are most effective at reducing Garmin's proprietary stress metric?")
    name_stress_change = plot_df.groupby('name')[['stress_pre_avg', 'stress_post_avg']].mean().reset_index()
    name_stress_change['stress_pct_change'] = (name_stress_change['stress_post_avg'] - name_stress_change['stress_pre_avg']) / name_stress_change['stress_pre_avg'] * 100
    fig5, ax3 = plt.subplots(figsize=(12, 6))
    name_stress_change = name_stress_change.sort_values('stress_pct_change', ascending=False)
    stress_colors = ['green' if val > 0 else 'red' for val in name_stress_change['stress_pct_change']]
    ax3.bar(name_stress_change['name'], name_stress_change['stress_pct_change'], color=stress_colors)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title("Stress Reduction (%) by Event/Intervention Name")
    ax3.set_xlabel("Event/Intervention")
    ax3.set_ylabel("Stress Reduction (%; higher is better)")
    ax3.set_xticklabels(name_stress_change['name'], rotation=45, ha='right')
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig5)
    
    # Sentiment Analysis Section
    st.write("### Sentiment Analysis: Reported Impact vs. Physiological Response")

    # Add explanation about sentiment analysis
    st.info("""
    This analysis shows the percentage of events/interventions that resulted in improved HRV for each reported impact category.
    - Higher percentages indicate that a greater proportion of events/interventions with that reported impact resulted in improved heart rate variability.
    - Improved HRV typically indicates better stress recovery and relaxation.
    """)

    # Ensure we have sentiment data
    if 'reported_impact' in plot_df.columns:
        # Filter out entries with "Unknown" reported impact
        filtered_plot_df = plot_df[plot_df['reported_impact'] != 'Unknown']
        
        if filtered_plot_df.empty:
            st.warning("No data with known impact feedback available for visualization. Please provide impact feedback in your annotations.")
        else:
            # Create a column indicating positive HRV change (improved HRV)
            filtered_plot_df['improved_hrv'] = filtered_plot_df['rmssd_change'] > 0
            
            # Group by reported impact and calculate percentage with improved HRV
            improved_hrv_pct = filtered_plot_df.groupby('reported_impact')['improved_hrv'].mean() * 100
            improved_hrv_pct = improved_hrv_pct.reset_index()
            improved_hrv_pct.columns = ['reported_impact', 'percent_improved']
            
            # Create figure for the chart
            fig_impact, ax = plt.subplots(figsize=(10, 6))
            
            # Plot reported impact vs percentage with improved HRV
            bars = ax.bar(improved_hrv_pct['reported_impact'], improved_hrv_pct['percent_improved'])
            ax.set_title('Percentage of Events/Interventions with Improved HRV by Reported Impact', fontsize=16)
            ax.set_xlabel('Reported Impact', fontsize=14)
            ax.set_ylabel('Percentage with Improved HRV (%)', fontsize=14)
            ax.set_ylim(0, 100)  # Set y-axis to range from 0 to 100%
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # Color bars
            for i, bar in enumerate(bars):
                bar.set_color('#77dd77')  # Green for all bars
            
            plt.tight_layout()
            st.pyplot(fig_impact)
    else:
        st.warning("Reported impact data is not available. Make sure to include impact feedback in your annotations.")
    
    # ----- Additional Temporal Patterns: Heatmap for Standardized Stress Reduction Magnitude -----
    # Compute global mean and std for stress_change from results_df
    mean_stress = results_df['stress_change'].mean()
    std_stress = results_df['stress_change'].std() if results_df['stress_change'].std() != 0 else 1e-6
    # Standardize stress_change (flip sign so that a decrease becomes positive)
    results_df['std_stress'] = -1 * (results_df['stress_change'] - mean_stress) / std_stress

    # Group by day of week and time of day to compute the average standardized stress reduction
    temporal_patterns = results_df.groupby(['day_of_week', 'time_of_day'])['std_stress'].mean().reset_index()

    # Optionally, define an order for the days of the week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    temporal_patterns['day_of_week'] = pd.Categorical(temporal_patterns['day_of_week'], categories=day_order, ordered=True)

    # Pivot the data so rows are days and columns are times
    temporal_pivot = temporal_patterns.pivot(index='day_of_week', columns='time_of_day', values='std_stress')
    # Reorder the columns to always correspond to morning, afternoon, evening, night
    time_cols = []
    for time in ["Morning", "Afternoon", "Evening", "Night"]:
        if time in temporal_pivot.columns:
            time_cols.append(time)
    temporal_pivot = temporal_pivot[time_cols]
    # Reindex to sort rows
    temporal_pivot = temporal_pivot.reindex(day_order)
    temporal_pivot = temporal_pivot.dropna(how="all")

    st.write("#### Heatmap: Average Standardized Stress Reduction by Day of Week and Time of Day")
    fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
    sns.heatmap(temporal_pivot, annot=True, cmap='viridis', fmt=".2f", ax=ax_heat)
    ax_heat.set_title("Standardized Stress Reduction by Day and Time")
    ax_heat.set_xlabel("Time of Day")
    ax_heat.set_ylabel("Day of Week")
    st.pyplot(fig_heat)
    
    # Define time points to visualize - not needed anymore with continuous data
    # time_points = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # Minutes after intervention - 1-minute intervals at the beginning
    
    # BBI Change Over Time - Using CONTINUOUS data with Altair
    st.write("### BBI Change Over Time After Events/Interventions")
    
    # Add explanation about BBI changes
    st.info("""
    **Note for participants:** 
    - **Positive BBI change (↑)** indicates an increase in the time interval between heartbeats, which typically means a more relaxed state, better recovery, and lower heart rate. This is generally beneficial for stress reduction.
    - **Negative BBI change (↓)** indicates a decrease in the time interval between heartbeats, which typically means a more activated state, higher heart rate, and potentially higher stress levels.
    """)
    
    # Check if BBI time data exists in session state to avoid re-running calculations
    if 'bbi_time_data' not in st.session_state or st.session_state.bbi_time_data is None:
        if not plot_df.empty:
            # Get BBI changes over time for each event/intervention
            bbi_time_data = pd.DataFrame()
            
            with st.spinner("Calculating continuous BBI changes over time..."):
                for event_name in plot_df['name'].unique():
                    # Get actual BBI changes over time for this event (60 minute window)
                    event_bbi_changes = calculate_bbi_continuous_change(username, event_name, 60)
                    
                    if not event_bbi_changes.empty:
                        # Save data for later display
                        event_bbi_changes['event_name'] = event_name
                        bbi_time_data = pd.concat([bbi_time_data, event_bbi_changes])
            
            # Save to session state to avoid recalculation
            st.session_state.bbi_time_data = bbi_time_data
        else:
            st.session_state.bbi_time_data = pd.DataFrame()
    
    # Use the data from session state for visualization
    bbi_time_data = st.session_state.bbi_time_data
    
    if not bbi_time_data.empty:
        # Create an interactive Altair chart with hover tooltips
        # Format the data for Altair
        bbi_time_data['minutes_after'] = bbi_time_data['minutes_after'].astype(float)
        bbi_time_data['bbi_change'] = bbi_time_data['bbi_change'].astype(float)
        
        # Create a unique label combining event name and instance ID (safely checking if intervention_id exists)
        if 'intervention_id' in bbi_time_data.columns:
            # Only for display in tooltips
            bbi_time_data['event_instance'] = bbi_time_data.apply(
                lambda row: f"{row['event_name']} #{row['intervention_id']}", 
                axis=1
            )
        else:
            # Fallback: just use event name
            bbi_time_data['event_instance'] = bbi_time_data['event_name']
            
        # Calculate average values for each event type at regular intervals (for smoother visualization)
        # First, round minutes to the nearest 0.5 to reduce noise but keep high granularity
        bbi_time_data['minutes_rounded'] = (bbi_time_data['minutes_after'] * 2).round() / 2
        avg_bbi_data = bbi_time_data.groupby(['event_name', 'minutes_rounded'])['bbi_change'].mean().reset_index()
        avg_bbi_data['bbi_change_formatted'] = avg_bbi_data['bbi_change'].map(lambda x: f"{x:.2f}%")
        
        # Get a count of instances per event for tooltips
        event_counts = bbi_time_data.groupby('event_name')['event_instance'].nunique().to_dict()
        avg_bbi_data['instance_count'] = avg_bbi_data['event_name'].map(event_counts)
        avg_bbi_data['event_name_with_count'] = avg_bbi_data.apply(
            lambda row: f"{row['event_name']} (avg of {row['instance_count']} instances)" 
            if row['instance_count'] > 1 else row['event_name'], 
            axis=1
        )
        
        # Rename minutes_rounded to minutes_after for consistency
        avg_bbi_data = avg_bbi_data.rename(columns={'minutes_rounded': 'minutes_after'})
        
        # Create a line chart with continuous data
        bbi_chart = alt.Chart(avg_bbi_data).mark_line(
            point=False,  # No points for smoother continuous look
            interpolate='linear', 
            strokeWidth=2.5  # Slightly thicker lines for better visibility
        ).encode(
            x=alt.X('minutes_after:Q', 
                   title='Minutes After Event/Intervention (0 = End of Intervention)',
                   scale=alt.Scale(domain=[0, 60])),  # Set x-axis to show 60 minute range
            y=alt.Y('bbi_change:Q', title='BBI Change (%)'),
            color=alt.Color('event_name:N', title='Event/Intervention'),
            tooltip=[
                alt.Tooltip('event_name_with_count:N', title='Event/Intervention'),
                alt.Tooltip('minutes_after:Q', title='Minutes After'),
                alt.Tooltip('bbi_change_formatted:N', title='Avg BBI Change'),
                alt.Tooltip('instance_count:Q', title='Number of Instances')
            ]
        ).properties(
            width=700,
            height=400,
            title='Average BBI Change Over Time After Events/Interventions (60 min)'
        )
        
        # Add a horizontal rule at y=0
        zero_rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='black').encode(y='y')
        
        # Combine the chart and the rule
        final_chart = alt.layer(bbi_chart, zero_rule).interactive()
        
        # Display the chart
        st.altair_chart(final_chart, use_container_width=True)
        
        # Optionally display the data table with average values
        if st.checkbox("Show BBI change data table", value=False):
            display_data = avg_bbi_data[['event_name', 'minutes_after', 'bbi_change_formatted', 'instance_count']]
            display_data.columns = ['Event/Intervention', 'Minutes After', 'Avg BBI Change', 'Number of Instances']
            st.dataframe(display_data)
    else:
        st.warning("No data available for BBI time-series analysis.")
    
    # HRV Change Over Time - Using CONTINUOUS data with Altair
    st.write("### HRV Change Over Time After Events/Interventions")
    
    # Add explanation about HRV changes
    st.info("""
    **Note for participants:** 
    - **Positive HRV change (↑)** indicates increased heart rate variability, which typically means better stress recovery, improved relaxation, and enhanced autonomic nervous system balance. This is generally beneficial.
    - **Negative HRV change (↓)** indicates decreased heart rate variability, which typically means higher stress, reduced recovery capacity, and potential autonomic imbalance.
    """)
    
    # Check if HRV time data exists in session state to avoid re-running calculations
    if 'hrv_time_data' not in st.session_state or st.session_state.hrv_time_data is None:
        if not plot_df.empty:
            # Get HRV changes over time for each event/intervention
            hrv_time_data = pd.DataFrame()
            
            with st.spinner("Calculating continuous HRV changes over time..."):
                for event_name in plot_df['name'].unique():
                    # Get actual HRV changes over time for this event (60 minute window)
                    event_hrv_changes = calculate_hrv_continuous_change(username, event_name, 60)
                    
                    if not event_hrv_changes.empty:
                        # Save data for later display
                        event_hrv_changes['event_name'] = event_name
                        hrv_time_data = pd.concat([hrv_time_data, event_hrv_changes])
            
            # Save to session state to avoid recalculation
            st.session_state.hrv_time_data = hrv_time_data
        else:
            st.session_state.hrv_time_data = pd.DataFrame()
    
    # Use the data from session state for visualization
    hrv_time_data = st.session_state.hrv_time_data
    
    if not hrv_time_data.empty:
        # Create an interactive Altair chart with hover tooltips
        # Format the data for Altair
        hrv_time_data['minutes_after'] = hrv_time_data['minutes_after'].astype(float)
        hrv_time_data['rmssd_change'] = hrv_time_data['rmssd_change'].astype(float)
        
        # Create a unique label combining event name and instance ID (safely checking if intervention_id exists)
        if 'intervention_id' in hrv_time_data.columns:
            # Only for display in tooltips
            hrv_time_data['event_instance'] = hrv_time_data.apply(
                lambda row: f"{row['event_name']} #{row['intervention_id']}", 
                axis=1
            )
        else:
            # Fallback: just use event name
            hrv_time_data['event_instance'] = hrv_time_data['event_name']
            
        # Calculate average values for each event type at regular intervals (for smoother visualization)
        # First, round minutes to the nearest 0.5 to reduce noise but keep high granularity
        hrv_time_data['minutes_rounded'] = (hrv_time_data['minutes_after'] * 2).round() / 2
        avg_hrv_data = hrv_time_data.groupby(['event_name', 'minutes_rounded'])['rmssd_change'].mean().reset_index()
        avg_hrv_data['rmssd_change_formatted'] = avg_hrv_data['rmssd_change'].map(lambda x: f"{x:.2f}%")
        
        # Get a count of instances per event for tooltips
        event_counts = hrv_time_data.groupby('event_name')['event_instance'].nunique().to_dict()
        avg_hrv_data['instance_count'] = avg_hrv_data['event_name'].map(event_counts)
        avg_hrv_data['event_name_with_count'] = avg_hrv_data.apply(
            lambda row: f"{row['event_name']} (avg of {row['instance_count']} instances)" 
            if row['instance_count'] > 1 else row['event_name'], 
            axis=1
        )
        
        # Rename minutes_rounded to minutes_after for consistency
        avg_hrv_data = avg_hrv_data.rename(columns={'minutes_rounded': 'minutes_after'})
        
        # Create a line chart with continuous data
        hrv_chart = alt.Chart(avg_hrv_data).mark_line(
            point=False,  # No points for smoother continuous look
            interpolate='linear',
            strokeWidth=2.5  # Slightly thicker lines for better visibility
        ).encode(
            x=alt.X('minutes_after:Q', 
                   title='Minutes After Event/Intervention (0 = End of Intervention)',
                   scale=alt.Scale(domain=[0, 60])),  # Set x-axis to show 60 minute range
            y=alt.Y('rmssd_change:Q', title='HRV Change (%)'),
            color=alt.Color('event_name:N', title='Event/Intervention'),
            tooltip=[
                alt.Tooltip('event_name_with_count:N', title='Event/Intervention'),
                alt.Tooltip('minutes_after:Q', title='Minutes After'),
                alt.Tooltip('rmssd_change_formatted:N', title='Avg HRV Change'),
                alt.Tooltip('instance_count:Q', title='Number of Instances')
            ]
        ).properties(
            width=700,
            height=400,
            title='Average HRV Change Over Time After Events/Interventions (60 min)'
        )
        
        # Add a horizontal rule at y=0
        zero_rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='black').encode(y='y')
        
        # Combine the chart and the rule
        final_chart = alt.layer(hrv_chart, zero_rule).interactive()
        
        # Display the chart
        st.altair_chart(final_chart, use_container_width=True)
        
        # Optionally display the data table with average values
        if st.checkbox("Show HRV change data table", value=False):
            display_data = avg_hrv_data[['event_name', 'minutes_after', 'rmssd_change_formatted', 'instance_count']]
            display_data.columns = ['Event/Intervention', 'Minutes After', 'Avg HRV Change', 'Number of Instances']
            st.dataframe(display_data)
    else:
        st.warning("No data available for HRV time-series analysis.")
    
    # Display category impact table
    display_category_impact_table(plot_df)
    
    return results_df

def upload_annotations(username: str):
    """
    Allows uploading event/intervention annotations from a CSV/Excel file.
    Saves them exactly like manual entries.
    """
    st.subheader("Upload Annotations")
    st.markdown("Upload a CSV or Excel file with your events and interventions data.")
    st.info("All uploaded events and interventions will be associated with your account for feature extraction purposes.")
    clear_existing = st.checkbox("Clear all existing annotations before import", help="If checked, all your existing events and interventions will be deleted before importing new ones")
    uploaded_file = st.file_uploader("Choose an annotation file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df.columns = [' '.join(col.split()).strip() if isinstance(col, str) else col for col in df.columns]
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            expected_columns = {
                'Event Type': 'Was this an intervention or an event?',
                'Start Time': 'Start time',
                'End Time': 'End time',
                'Name': 'Event/Intervention name',
                'Notes': 'Please note some descriptions that reflect how you felt during that time',
                'Impact Feedback': 'How do you think this event or intervention impacted your stress?',
                'Date': 'Date',
                'Participant': 'Participant #'
            }
            found_columns = {}
            for field, expected_col in expected_columns.items():
                for col in df.columns:
                    if isinstance(col, str) and expected_col.lower() in col.lower():
                        found_columns[field] = col
                        break
            st.subheader("Map Columns to Database Fields")
            st.markdown("Please verify these column mappings are correct:")
            col_mapping = {}
            for field, description in expected_columns.items():
                default_col = found_columns.get(field, "None")
                help_text = f"Column that contains {description}"
                selected_col = st.selectbox(
                    f"Column for {field}:",
                    options=["None"] + list(df.columns),
                    index=0 if default_col == "None" else list(df.columns).index(default_col) + 1,
                    help=help_text
                )
                col_mapping[field] = None if selected_col == "None" else selected_col
            time_format = st.selectbox("Time format in your data:", options=["12-hour (e.g., 1:30 PM)", "24-hour (e.g., 13:30)"], index=0)
            if st.button("Import Annotations", key="import_button", type="primary"):
                if col_mapping['Start Time'] is None or col_mapping['End Time'] is None:
                    st.error("Start Time and End Time must be mapped.")
                    return
                if col_mapping['Name'] is None:
                    st.error("The 'Name' field must be mapped to properly identify events/interventions.")
                    return
                if clear_existing:
                    with get_rds_connection() as conn:
                        with conn.cursor() as cursor:
                            st.warning("Clearing all existing annotations...")
                            cursor.execute("DELETE FROM events WHERE username = %s", (username,))
                            cursor.execute("DELETE FROM interventions WHERE username = %s", (username,))
                            conn.commit()
                            st.success("Existing annotations cleared.")
                progress_bar = st.progress(0)
                total_rows = len(df)
                success_count = 0
                event_count = 0
                intervention_count = 0
                error_count = 0
                for idx, row in df.iterrows():
                    try:
                        progress_bar.progress((idx + 1) / total_rows)
                        user_col = username
                        event_type_text = ""
                        if col_mapping['Event Type'] and not pd.isna(row[col_mapping['Event Type']]):
                            event_type_text = str(row[col_mapping['Event Type']]).strip().lower()
                        is_intervention = 'intervention' in event_type_text
                        table_name = 'interventions' if is_intervention else 'events'
                        activity_name = ""
                        if col_mapping['Name'] and not pd.isna(row[col_mapping['Name']]):
                            activity_name = str(row[col_mapping['Name']]).strip()
                        if not activity_name:
                            st.warning(f"Row {idx+1}: Skipping because it has no activity name.")
                            error_count += 1
                            continue
                        try:
                            date_obj = datetime.today().date()
                            if col_mapping['Date'] and not pd.isna(row[col_mapping['Date']]):
                                date_str = str(row[col_mapping['Date']])
                                try:
                                    date_obj = pd.to_datetime(date_str).date()
                                except:
                                    st.warning(f"Could not parse date '{date_str}' in row {idx+1}. Using today's date.")
                            start_str = str(row[col_mapping['Start Time']])
                            end_str = str(row[col_mapping['End Time']])
                            try:
                                if time_format == "12-hour (e.g., 1:30 PM)":
                                    start_time = pd.to_datetime(f"{date_obj} {start_str}")
                                    end_time = pd.to_datetime(f"{date_obj} {end_str}")
                                else:
                                    start_time = pd.to_datetime(f"{date_obj} {start_str}")
                                    end_time = pd.to_datetime(f"{date_obj} {end_str}")
                            except:
                                start_time = pd.to_datetime(f"{date_obj} {start_str}", errors='coerce')
                                end_time = pd.to_datetime(f"{date_obj} {end_str}", errors='coerce')
                            if pd.isna(start_time) or pd.isna(end_time):
                                raise ValueError(f"Unable to parse start/end time from row {idx+1}: {start_str} / {end_str}")
                            start_ms = int(start_time.timestamp() * 1000)
                            end_ms = int(end_time.timestamp() * 1000)
                            notes = ""
                            if col_mapping['Notes'] and not pd.isna(row[col_mapping['Notes']]):
                                notes = str(row[col_mapping['Notes']]).strip()
                            impact_feedback = ""
                            if col_mapping['Impact Feedback'] and not pd.isna(row[col_mapping['Impact Feedback']]):
                                impact_feedback = str(row[col_mapping['Impact Feedback']]).strip()
                            if col_mapping['Participant'] and not pd.isna(row[col_mapping['Participant']]):
                                participant = str(row[col_mapping['Participant']]).strip()
                                if participant:
                                    notes = notes + f" [Participant: {participant}]" if notes else f"[Participant: {participant}]"
                            st.write(f"Row {idx+1}: Adding to {table_name} table - Name: '{activity_name}', Type: '{'Intervention' if is_intervention else 'Event'}'")
                            from sql_utils import record_event_in_database
                            record_event_in_database(
                                user=username,
                                start_time=start_ms,
                                end_time=end_ms,
                                event_name=activity_name,
                                var_name=table_name,
                                category="Intervention" if is_intervention else "Event",
                                notes=notes,
                                impact_feedback=impact_feedback,
                            )

                            success_count += 1
                            if is_intervention:
                                intervention_count += 1
                            else:
                                event_count += 1
                        except Exception as time_error:
                            st.warning(f"Time parsing error at row {idx+1}: {str(time_error)}. Skipping.")
                            error_count += 1
                            continue
                    except Exception as row_error:
                        st.warning(f"Error processing row {idx+1}: {str(row_error)}")
                        error_count += 1
                        continue
                if success_count > 0:
                    st.success(f"Successfully imported {success_count} annotations ({event_count} events, {intervention_count} interventions). Encountered {error_count} errors.")
                    with get_rds_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT COUNT(*) FROM events WHERE username = %s", (username,))
                            total_events = cursor.fetchone()[0]
                            cursor.execute("SELECT COUNT(*) FROM interventions WHERE username = %s", (username,))
                            total_interventions = cursor.fetchone()[0]
                            st.write(f"You now have {total_events} events and {total_interventions} interventions in the database.")
                else:
                    st.error("Failed to import any annotations. Please check your column mappings.")
        except Exception as e:
            st.error(f"Error reading or processing file: {e}")

def debug_tables(username: str):
    """
    Debug function to check contents of the database tables.
    """
    st.subheader("Database Table Contents")
    with get_rds_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute("SELECT * FROM events WHERE username = %s", (username,))
                events_data = cursor.fetchall()
                if events_data:
                    events_cols = [desc[0] for desc in cursor.description]
                    st.write("Events table data:")
                    st.dataframe(pd.DataFrame(events_data, columns=events_cols))
                else:
                    st.warning("No events found for user")
            except Exception as e:
                st.error(f"Error querying events table: {str(e)}")
            try:
                cursor.execute("SELECT * FROM interventions WHERE username = %s", (username,))
                int_data = cursor.fetchall()
                if int_data:
                    int_cols = [desc[0] for desc in cursor.description]
                    st.write("Interventions table data:")
                    st.dataframe(pd.DataFrame(int_data, columns=int_cols))
                else:
                    st.warning("No interventions found for user")
            except Exception as e:
                st.error(f"Error querying interventions table: {str(e)}")
            for table in ['events', 'interventions']:
                try:
                    cursor.execute(f"DESCRIBE {table}")
                    structure = cursor.fetchall()
                    st.write(f"{table} table structure:")
                    st.dataframe(pd.DataFrame(structure, columns=['Field', 'Type', 'Null', 'Key', 'Default', 'Extra']))
                except Exception as e:
                    st.error(f"Error describing {table} table: {str(e)}")

def recategorize_annotations(username: str):
    """Fixes event/intervention categorization."""
    st.subheader("Recategorize Events/Interventions")
    with get_rds_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, name, category FROM events WHERE username = %s", (username,))
            events = cursor.fetchall()
            cursor.execute("SELECT id, name, category FROM interventions WHERE username = %s", (username,))
            interventions = cursor.fetchall()
            if not events and not interventions:
                st.warning("No events or interventions found to recategorize.")
                return
            if events:
                st.write("### Current Events")
                for event in events:
                    event_id, event_name, event_category = event
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{event_name}** (ID: {event_id})")
                    with col2:
                        if st.button(f"Make Intervention #{event_id}", key=f"ev_{event_id}"):
                            try:
                                cursor.execute("SELECT * FROM events WHERE id = %s", (event_id,))
                                record = cursor.fetchone()
                                cols = [desc[0] for desc in cursor.description]
                                record_dict = {col: val for col, val in zip(cols, record)}
                                record_dict['category'] = 'Intervention'
                                placeholders = ", ".join(["%s"] * len(record_dict))
                                columns = ", ".join(record_dict.keys())
                                cursor.execute(f"INSERT INTO interventions ({columns}) VALUES ({placeholders})", tuple(record_dict.values()))
                                cursor.execute("DELETE FROM events WHERE id = %s", (event_id,))
                                conn.commit()
                                st.success(f"Moved '{event_name}' to interventions.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error recategorizing: {e}")
            if interventions:
                st.write("### Current Interventions")
                for intervention in interventions:
                    int_id, int_name, int_category = intervention
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{int_name}** (ID: {int_id})")
                    with col2:
                        if st.button(f"Make Event #{int_id}", key=f"int_{int_id}"):
                            try:
                                cursor.execute("SELECT * FROM interventions WHERE id = %s", (int_id,))
                                record = cursor.fetchone()
                                cols = [desc[0] for desc in cursor.description]
                                record_dict = {col: val for col, val in zip(cols, record)}
                                record_dict['category'] = 'Event'
                                placeholders = ", ".join(["%s"] * len(record_dict))
                                columns = ", ".join(record_dict.keys())
                                cursor.execute(f"INSERT INTO events ({columns}) VALUES ({placeholders})", tuple(record_dict.values()))
                                cursor.execute("DELETE FROM interventions WHERE id = %s", (int_id,))
                                conn.commit()
                                st.success(f"Moved '{int_name}' to events.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error recategorizing: {e}")

# Define intervention categories
INTERVENTION_CATEGORIES = {
    "Physical Activity (Cardio)": [
        "run", "running", "jog", "jogging", "cycling", "bike", "spin class",
        "hiit", "cardio", "treadmill", "exercise session", "gym session",
        "swimming", "aerobic", "dance", "zumba", "workout", "fitness"
    ],
    "Physical Activity (Non-Cardio)": [
        "walk", "walking", "yoga", "stretch", "light exercise", "light yoga",
        "pilates", "tai chi", "strength", "weights", "resistance"
    ],
    "Social Media Consumption": [
        "scrolling", "social media", "facebook", "instagram", "tiktok",
        "twitter", "youtube", "screen time", "browsing", "reddit", "linkedin"
    ],
    "Nap/Rest": [
        "nap", "rest", "sleep", "break", "relaxation", "quiet time", "recharge"
    ],
    "Eating/Drinking": [
        "eating", "drinking", "coffee", "lunch", "breakfast", "dinner", "meal",
        "snack", "tea", "water", "food", "cooking", "brunch"
    ],
    "Social Interaction": [
        "call", "social", "meeting", "friends", "conversation", "hangout",
        "chat", "discussion", "talk", "socializing", "family", "dinner with", "lunch with"
    ],
    "Journaling/Writing": [
        "journal", "journaling", "writing", "gratitude", "diary", "planning", "reflection"
    ],
    "Reading": [
        "read", "reading", "book", "novel", "fiction", "article"
    ],
    "Other": []
}

def categorize_intervention(name: str) -> str:
    """
    Categorize an intervention based on its name.
    """
    name_lower = name.lower()
    for category, keywords in INTERVENTION_CATEGORIES.items():
        if any(keyword in name_lower for keyword in keywords):
            return category
    return "Other"

def display_category_impact_table(results_df: pd.DataFrame):
    """
    Display a table showing intervention categories, their frequency, and impact at different time intervals.
    """
    if results_df.empty:
        st.warning("No data available for category analysis")
        return
    
    # Make a copy to avoid modifying the original
    df = results_df.copy()
    
    # Categorize interventions
    df['intervention_category'] = df['name'].apply(categorize_intervention)
    
    # Calculate category statistics
    category_stats = []
    for category in sorted(df['intervention_category'].unique()):
        category_df = df[df['intervention_category'] == category]
        frequency = len(category_df)
        frequency_pct = (frequency / len(df)) * 100
        
        # Get HRV time data from session state if available
        calculate_different_windows = False
        hrv_time_data = pd.DataFrame()
        
        if 'hrv_time_data' in st.session_state and st.session_state.hrv_time_data is not None:
            hrv_time_data = st.session_state.hrv_time_data
            
            # If we have time data and events from this category are included,
            # we'll calculate impacts for different time windows
            if not hrv_time_data.empty:
                event_names = category_df['name'].unique()
                relevant_hrv_data = hrv_time_data[hrv_time_data['event_name'].isin(event_names)]
                
                if not relevant_hrv_data.empty:
                    calculate_different_windows = True
        
        # Calculate base impact using the existing metrics in results_df
        if 'rmssd_change' in category_df.columns:
            avg_impact = category_df['rmssd_change'].mean()
        elif 'rmssd_pct_change' in category_df.columns:
            avg_impact = category_df['rmssd_pct_change'].mean()
        elif 'rmssd_pre_avg' in category_df.columns and 'rmssd_post_avg' in category_df.columns:
            # Calculate percentage change directly
            avg_pre = category_df['rmssd_pre_avg'].mean()
            avg_post = category_df['rmssd_post_avg'].mean()
            if avg_pre != 0:  # Avoid division by zero
                avg_impact = ((avg_post - avg_pre) / avg_pre) * 100
            else:
                avg_impact = None
        else:
            avg_impact = None
            
        # Default: use base impact for all time windows
        avg_impact_15 = avg_impact
        avg_impact_30 = avg_impact
        avg_impact_60 = avg_impact
        
        # If we have continuous time data, calculate different impacts for each window
        if calculate_different_windows:
            # For each event in this category, get the HRV changes at the time points
            event_names = category_df['name'].unique()
            
            # Filter HRV time data for events in this category
            category_hrv_data = hrv_time_data[hrv_time_data['event_name'].isin(event_names)]
            
            # Calculate impacts at different time windows
            if not category_hrv_data.empty:
                # 15-minute window
                window_15 = category_hrv_data[(category_hrv_data['minutes_after'] >= 0) & 
                                           (category_hrv_data['minutes_after'] <= 15)]
                if not window_15.empty:
                    avg_impact_15 = window_15['rmssd_change'].mean()
                
                # 30-minute window
                window_30 = category_hrv_data[(category_hrv_data['minutes_after'] >= 0) & 
                                           (category_hrv_data['minutes_after'] <= 30)]
                if not window_30.empty:
                    avg_impact_30 = window_30['rmssd_change'].mean()
                
                # 60-minute window
                window_60 = category_hrv_data[(category_hrv_data['minutes_after'] >= 0) & 
                                           (category_hrv_data['minutes_after'] <= 60)]
                if not window_60.empty:
                    avg_impact_60 = window_60['rmssd_change'].mean()
                
                # Convert from decimal to percentage (for display consistency)
                avg_impact_15 = avg_impact_15 * 100 if avg_impact_15 is not None else None
                avg_impact_30 = avg_impact_30 * 100 if avg_impact_30 is not None else None
                avg_impact_60 = avg_impact_60 * 100 if avg_impact_60 is not None else None
        
        category_stats.append({
            'Intervention Category': category,
            'Frequency (%)': f"{frequency_pct:.1f}%",
            'Avg Impact (15 min)': f"{avg_impact_15:.1f}%" if avg_impact_15 is not None and not pd.isna(avg_impact_15) else "N/A",
            'Avg Impact (30 min)': f"{avg_impact_30:.1f}%" if avg_impact_30 is not None and not pd.isna(avg_impact_30) else "N/A", 
            'Avg Impact (60 min)': f"{avg_impact_60:.1f}%" if avg_impact_60 is not None and not pd.isna(avg_impact_60) else "N/A"
        })
    
    # Create DataFrame for display
    stats_df = pd.DataFrame(category_stats)
    
    # Display the table
    st.write("### Summary of Event/Intervention Categories Impact on HRV")
    st.write("This table shows how different categories of events/interventions affect HRV over time.")
    st.table(stats_df)
    
    st.write("**Note:** Positive HRV changes indicate better stress recovery and relaxation, while negative changes may indicate higher stress.")

# ------------------ MAIN APP FUNCTION ------------------ #
def run_stepper_extraction():
    """
    Main entry function for the Feature Extraction tool.
    
    This tool extracts physiological features from wearable data in relation to events and interventions,
    computes pre/post changes, calculates a composite impact score, and uses exploratory ML to identify patterns.
    """
    st.title("Feature Extractions (Admin)")
    st.markdown("""
        *This tool extracts physiological features from wearable data in relation to events and interventions,
        and uses exploratory ML to identify patterns for personalized stress management.*
    """)
    username = st.session_state.get('username') or st.session_state.get('user')
    if not username:
        st.warning("Please log in to use this feature")
        return
    st.success(f"Processing data for: {username}")
    tab1, tab2, tab3 = st.tabs(["Extract Features", "Upload Annotations", "Manage Annotations"])
    
    with tab1:
        st.subheader("Analysis Options")
        minutes_window = st.slider("Minutes to analyze before/after events:", 
                               min_value=5, 
                               max_value=120, 
                               value=5,
                               help="Number of minutes to analyze before and after each event or intervention")
        
        # Convert minutes to hours for the analysis
        hours_window = minutes_window / 60.0
        st.markdown("""
        This feature analyzes the physiological impact of events and interventions by comparing measurements
        before and after each occurrence. It also uses exploratory ML (unsupervised clustering) to automatically
        extract physiologically relevant features and identify patterns that predict whether specific
        interventions or events yield positive or negative outcomes in reducing stress.
        """)
        
        # Initialize session state for results if not present
        if 'impact_results' not in st.session_state:
            st.session_state.impact_results = None
        if 'last_minutes_window' not in st.session_state:
            st.session_state.last_minutes_window = minutes_window
            
        # Check if we need to update results
        should_update = (st.session_state.last_minutes_window != minutes_window or 
                        st.session_state.impact_results is None)
        
        if st.button("Extract Features") or should_update:
            with st.spinner("Analyzing physiological data..."):
                try:
                    results = analyze_physiological_impact(username, hours_window)
                    if results is not None and not results.empty:
                        st.session_state.impact_results = results
                        st.session_state.last_minutes_window = minutes_window
                        visualize_analysis(results, username)
                        csv = results.to_csv(index=False)
                        st.download_button(label="Download Analysis as CSV", data=csv,
                                           file_name=f"{username}_feature_extraction.csv", mime="text/csv")
                    else:
                        st.warning("No data available for analysis. Please ensure you have both events/interventions and physiological data.")
                        st.subheader("Database Tables for User")
                        with get_rds_connection() as conn:
                            with conn.cursor() as cursor:
                                for tbl in ['events', 'interventions']:
                                    cursor.execute(f"""
                                        SELECT COUNT(*) 
                                        FROM information_schema.tables
                                        WHERE table_name = '{tbl}'
                                    """)
                                    if cursor.fetchone()[0] > 0:
                                        cursor.execute(f"SELECT * FROM {tbl} WHERE username = %s LIMIT 5", (username,))
                                        data = cursor.fetchall()
                                        if data:
                                            st.write(f"{tbl.capitalize()} data found:")
                                            columns = [desc[0] for desc in cursor.description]
                                            st.dataframe(pd.DataFrame(data, columns=columns))
                                        else:
                                            st.warning(f"No {tbl} found for this user")
                                    else:
                                        st.warning(f"{tbl.capitalize()} table does not exist")
                                for metric in ['rmssd', 'bbi', 'steps', 'stress']:
                                    metric_table = f"{metric}_{username}"
                                    cursor.execute(f"""
                                        SELECT COUNT(*) 
                                        FROM information_schema.tables
                                        WHERE table_name = '{metric_table}'
                                    """)
                                    if cursor.fetchone()[0] > 0:
                                        cursor.execute(f"SELECT COUNT(*) FROM {metric_table}")
                                        count = cursor.fetchone()[0]
                                        st.write(f"{metric.upper()} data: {count} records")
                                    else:
                                        st.warning(f"No {metric.upper()} data table found for this user")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        elif st.session_state.impact_results is not None:
            # Display existing results if available
            visualize_analysis(st.session_state.impact_results, username)
            csv = st.session_state.impact_results.to_csv(index=False)
            st.download_button(label="Download Analysis as CSV", data=csv,
                               file_name=f"{username}_feature_extraction.csv", mime="text/csv")
    
    with tab2:
        upload_annotations(username)
    
    with tab3:
        recategorize_annotations(username)
