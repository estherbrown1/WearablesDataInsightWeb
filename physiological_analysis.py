import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
from datetime import datetime, timedelta
import re
from sql_utils import get_rds_connection
import streamlit as st

# ML imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import VAR_DICT from your visualization module (adjust the path as needed)
from visualization_page import VAR_DICT  

# ------------------ NEW: CLEAR OLD ROWS FUNCTION ------------------ #
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

# ------------------ 1. HELPER FUNCTIONS ------------------ #

def get_table_data(username: str, table_name: str) -> pd.DataFrame:
    """
    Fetches data from the specified table (either 'events' or 'interventions')
    for a given username.
    """
    try:
        with get_rds_connection() as conn:
            with conn.cursor() as cursor:
                # First, check if the table exists
                cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM information_schema.tables
                    WHERE table_name = '{table_name}'
                """)
                if cursor.fetchone()[0] == 0:
                    st.warning(f"Table {table_name} does not exist")
                    return pd.DataFrame()
                
                # Get actual column names
                cursor.execute(f"DESCRIBE {table_name}")
                columns = [row[0] for row in cursor.fetchall()]
                
                # Build a query based on existing columns
                select_cols = []
                for col in ["id", "username", "name", "start_time", "end_time", "category", "notes", "impact_feedback"]:
                    if col in columns:
                        select_cols.append(col)
                
                if not select_cols:
                    st.warning(f"No usable columns found in {table_name}")
                    return pd.DataFrame()
                
                # Execute the query with proper parameters
                query = f"SELECT {', '.join(select_cols)} FROM {table_name} WHERE username = %s"
                cursor.execute(query, (username,))
                rows = cursor.fetchall()
                
                if not rows:
                    st.warning(f"No data found in {table_name} for user {username}")
                    return pd.DataFrame()
                
                # Create a DataFrame with the results
                df = pd.DataFrame(rows, columns=select_cols)
                
                # Convert timestamp columns (assumed stored in milliseconds)
                for col in ["start_time", "end_time"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], unit='ms')
                
                # Clean up the "name" column
                if "name" in df.columns:
                    df["name"] = df["name"].fillna("Unspecified")
                else:
                    df["name"] = "Unspecified"
                
                # Set instance_type based on the table
                df["instance_type"] = "Event" if table_name == "events" else "Intervention"
                
                # Ensure category is set appropriately
                if "category" in df.columns:
                    df["category"] = "Event" if table_name == "events" else "Intervention"
                else:
                    df["category"] = "Event" if table_name == "events" else "Intervention"
                
                # Fill missing columns with defaults
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
    Includes timezone validation with improved type handling.
    """
    try:
        # Display the event times for reference
        st.write(f"Analyzing {metric_name} data around event time: {pre_start.strftime('%Y-%m-%d %H:%M:%S')} to {post_end.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Convert timestamps to milliseconds for database queries
        pre_start_ms = int(pre_start.timestamp() * 1000)
        pre_end_ms = int(pre_end.timestamp() * 1000)
        post_start_ms = int(post_start.timestamp() * 1000)
        post_end_ms = int(post_end.timestamp() * 1000)
        
        # Find the actual table name and column names in the database
        with get_rds_connection() as conn:
            with conn.cursor() as cursor:
                # Look for the table with the given username and metric
                table_name = f"{username}_{metric_name}"
                
                # Check if the table exists
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_name = %s
                """, (table_name,))
                
                if cursor.fetchone()[0] == 0:
                    st.warning(f"Table {table_name} not found.")
                    return {'percent_change': 0.0, 'pre_avg': None, 'post_avg': None, 'data_available': False}
                
                # Get columns from the table
                cursor.execute(f"SHOW COLUMNS FROM {table_name}")
                columns = [col[0] for col in cursor.fetchall()]
                
                # Check if unix_timestamp_cleaned is present
                if 'unix_timestamp_cleaned' not in columns:
                    st.warning(f"Column unix_timestamp_cleaned not found in {table_name}.")
                    return {'percent_change': 0.0, 'pre_avg': None, 'post_avg': None, 'data_available': False}
                
                # Determine the value column based on the metric
                value_column = None
                # Try to use the canonical value column from VAR_DICT
                try:
                    from visualization_page import VAR_DICT
                    var_name = VAR_DICT.get(metric_name, metric_name)
                    if var_name in columns:
                        value_column = var_name
                except (ImportError, KeyError):
                    pass
                
                # If that fails, try common names for the specific metric
                if not value_column:
                    if metric_name == 'stress' and 'stressLevel' in columns:
                        value_column = 'stressLevel'
                    elif metric_name in ['bbi', 'rmssd', 'steps'] and metric_name in columns:
                        value_column = metric_name
                    elif metric_name == 'steps' and 'totalSteps' in columns:
                        value_column = 'totalSteps'
                
                if not value_column:
                    st.warning(f"No suitable value column found in {table_name}.")
                    return {'percent_change': 0.0, 'pre_avg': None, 'post_avg': None, 'data_available': False}
                
                # Check for timezone issues by comparing first and last timestamps safely
                if 'timestamp_cleaned' in columns:
                    cursor.execute(f"""
                        SELECT MIN(unix_timestamp_cleaned), MAX(unix_timestamp_cleaned)
                        FROM {table_name}
                        WHERE unix_timestamp_cleaned > 0
                    """)
                    min_ms, max_ms = cursor.fetchone()
                    
                    # Safely convert timestamps to datetime objects
                    from datetime import datetime
                    try:
                        # Ensure numeric type for timestamps
                        if min_ms is not None and max_ms is not None:
                            min_ms = float(min_ms) if isinstance(min_ms, str) else min_ms
                            max_ms = float(max_ms) if isinstance(max_ms, str) else max_ms
                            
                            min_from_ms = datetime.fromtimestamp(min_ms/1000)
                            max_from_ms = datetime.fromtimestamp(max_ms/1000)
                            
                            st.write("Timezone verification:")
                            st.write(f"  Unix timestamp range: {min_ms} to {max_ms}")
                            st.write(f"  Converted to datetime: {min_from_ms} to {max_from_ms}")
                            
                            # Get readable timestamp range
                            cursor.execute(f"""
                                SELECT MIN(timestamp_cleaned), MAX(timestamp_cleaned)
                                FROM {table_name}
                            """)
                            min_readable, max_readable = cursor.fetchone()
                            st.write(f"  Readable timestamp range: {min_readable} to {max_readable}")
                    except (TypeError, ValueError) as e:
                        st.warning(f"Error converting timestamps: {e}")
                
                # Query pre-window data
                cursor.execute(f"""
                    SELECT COUNT(*), AVG({value_column})
                    FROM {table_name}
                    WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                """, (pre_start_ms, pre_end_ms))
                pre_count, pre_avg = cursor.fetchone()
                
                # Query post-window data
                cursor.execute(f"""
                    SELECT COUNT(*), AVG({value_column})
                    FROM {table_name}
                    WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                """, (post_start_ms, post_end_ms))
                post_count, post_avg = cursor.fetchone()
                
                # Get a sample of pre-window data for verification
                if pre_count > 0:
                    cursor.execute(f"""
                        SELECT unix_timestamp_cleaned, {value_column}
                        {', timestamp_cleaned' if 'timestamp_cleaned' in columns else ''}
                        FROM {table_name}
                        WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                        ORDER BY unix_timestamp_cleaned LIMIT 3
                    """, (pre_start_ms, pre_end_ms))
                    pre_samples = cursor.fetchall()
                    st.write(f"Pre-window sample data ({pre_count} records):")
                    for sample in pre_samples:
                        if 'timestamp_cleaned' in columns:
                            st.write(f"  Unix: {sample[0]}, Value: {sample[1]}, Time: {sample[2]}")
                        else:
                            unix_dt = datetime.fromtimestamp(float(sample[0])/1000 if isinstance(sample[0], str) else sample[0]/1000)
                            st.write(f"  Unix: {sample[0]}, Value: {sample[1]}, Time: {unix_dt}")
                
                # Get a sample of post-window data for verification
                if post_count > 0:
                    cursor.execute(f"""
                        SELECT unix_timestamp_cleaned, {value_column}
                        {', timestamp_cleaned' if 'timestamp_cleaned' in columns else ''}
                        FROM {table_name}
                        WHERE unix_timestamp_cleaned BETWEEN %s AND %s
                        ORDER BY unix_timestamp_cleaned LIMIT 3
                    """, (post_start_ms, post_end_ms))
                    post_samples = cursor.fetchall()
                    st.write(f"Post-window sample data ({post_count} records):")
                    for sample in post_samples:
                        if 'timestamp_cleaned' in columns:
                            st.write(f"  Unix: {sample[0]}, Value: {sample[1]}, Time: {sample[2]}")
                        else:
                            unix_dt = datetime.fromtimestamp(float(sample[0])/1000 if isinstance(sample[0], str) else sample[0]/1000)
                            st.write(f"  Unix: {sample[0]}, Value: {sample[1]}, Time: {unix_dt}")
                
                # Calculate percent change if data is available
                percent_change = 0.0
                if pre_avg is not None and post_avg is not None and pre_avg != 0 and pre_count > 0 and post_count > 0:
                    # Convert to float to avoid decimal.Decimal issues
                    pre_avg_float = float(pre_avg)
                    post_avg_float = float(post_avg)
                    percent_change = ((post_avg_float - pre_avg_float) / pre_avg_float) * 100
                    
                    # For display purposes
                    if metric_name in ["stress", "daily_heart_rate", "respiration"]:
                        direction = "decreased" if percent_change < 0 else "increased"
                        effect = "Positive" if percent_change < 0 else "Negative"
                    elif metric_name in ["bbi", "rmssd", "hrv"]:
                        direction = "increased" if percent_change > 0 else "decreased"
                        effect = "Positive" if percent_change > 0 else "Negative"
                    else:
                        direction = "changed"
                        effect = "Neutral"
                        
                    st.write(f"{metric_name} {direction} by {abs(percent_change):.2f}% ({effect} effect)")
                    
                    return {
                        'percent_change': float(percent_change),  # Explicitly cast to float
                        'pre_avg': float(pre_avg_float),
                        'post_avg': float(post_avg_float),
                        'data_available': True,
                        'is_positive': effect == "Positive"
                    }
                else:
                    st.warning(f"Insufficient data to calculate {metric_name} change. Pre-count: {pre_count}, Post-count: {post_count}")
                    return {
                        'percent_change': 0.0,
                        'pre_avg': float(pre_avg) if pre_avg is not None else None,
                        'post_avg': float(post_avg) if post_avg is not None else None,
                        'data_available': False
                    }
    
    except Exception as e:
        st.error(f"Error analyzing {metric_name} change: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return {'percent_change': 0.0, 'pre_avg': None, 'post_avg': None, 'data_available': False}

def calculate_impact_score(df: pd.DataFrame) -> pd.Series:
    """Calculates an overall impact score based on physiological changes."""
    # Note: We weight stress reduction positively (i.e. a negative percent change in stress is good)
    weights = {'hrv_change': 0.3, 'bbi_change': 0.2, 'steps_change': 0.1, 'stress_change': -0.4}
    impact = pd.Series(0.0, index=df.index)
    for metric, weight in weights.items():
        if metric in df.columns:
            impact += df[metric] * weight
    return impact

# ------------------ 2. MAIN ANALYSIS FUNCTION ------------------ #

def analyze_physiological_impact(username: str, hours_window: int = 2) -> pd.DataFrame:
    """
    Analyzes physiological patterns around events and interventions.
    - 'username' column: logged-in user
    - 'name' column: the descriptive event name
    - 'category' column: "Event" or "Intervention"
    
    NOTE: For heart rate variability, use 'rmssd' (as defined in VAR_DICT)
    """
    st.info("Analyzing physiological patterns around events and interventions...")

    # Fetch from both tables
    events_df = get_table_data(username, 'events')
    interventions_df = get_table_data(username, 'interventions')

    st.write(f"Found {len(events_df)} events and {len(interventions_df)} interventions")
    
    if not events_df.empty:
        st.write("Sample of events data:")
        st.dataframe(events_df[['name', 'instance_type', 'start_time', 'end_time']].head())
    if not interventions_df.empty:
        st.write("Sample of interventions data:")
        st.dataframe(interventions_df[['name', 'instance_type', 'start_time', 'end_time']].head())

    if events_df.empty and interventions_df.empty:
        st.warning(f"No events or interventions found for user {username}")
        return pd.DataFrame()

    all_instances = pd.concat([events_df, interventions_df], ignore_index=True)
    st.write("Combined data for analysis:")
    st.dataframe(all_instances[['name', 'instance_type', 'start_time', 'end_time']].head(3))

    # Extract sentiment and reported impact
    all_instances['sentiment'] = all_instances['notes'].apply(extract_sentiment)
    all_instances['reported_impact'] = all_instances['impact_feedback'].apply(extract_reported_impact)

    # Time-based features
    all_instances['time_of_day'] = all_instances['start_time'].apply(
        lambda x: 'Morning' if 5 <= x.hour < 12 else 
                  'Afternoon' if 12 <= x.hour < 17 else 
                  'Evening' if 17 <= x.hour < 21 else 'Night'
    )
    all_instances['day_of_week'] = all_instances['start_time'].dt.day_name()

    results = []
    # IMPORTANT: Use the correct metric names – note we now use "rmssd" (from VAR_DICT) for HRV instead of "hrv"
    for _, instance in all_instances.iterrows():
        start_time = instance['start_time']
        end_time = instance['end_time']
        duration_mins = (end_time - start_time).total_seconds() / 60
        pre_window = (start_time - pd.Timedelta(hours=hours_window), start_time)
        post_window = (end_time, end_time + pd.Timedelta(hours=hours_window))

        metrics_analysis = {}
        pre_metrics = {}
        post_metrics = {}
        for metric in ['rmssd', 'bbi', 'steps', 'stress']:  # Note: "rmssd" is used for HRV now.
            metric_change = analyze_metric_change(username, metric,
                                                  pre_window[0], pre_window[1],
                                                  post_window[0], post_window[1])
            metrics_analysis[f'{metric}_change'] = metric_change['percent_change']
            metrics_analysis[f'{metric}_pre_avg'] = metric_change['pre_avg']
            metrics_analysis[f'{metric}_post_avg'] = metric_change['post_avg']
            pre_metrics[metric] = metric_change['pre_avg']
            post_metrics[metric] = metric_change['post_avg']

        # Determine stress reduction (a decrease in stress is positive)
        stress_reduction = False
        if pre_metrics['stress'] is not None and post_metrics['stress'] is not None:
            stress_reduction = pre_metrics['stress'] > post_metrics['stress']

        # Determine pre-stress state based on stress level
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
        results_df['impact_score'] = calculate_impact_score(results_df)
        st.write("Results data ready for visualization:")
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

# ------------------ 3. EXPLORATORY ML & VISUALIZATIONS ------------------ #

def exploratory_ml_analysis(results_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    st.subheader("Exploratory ML Analysis")
    if results_df.empty:
        st.warning("No data to analyze.")
        return results_df

    feature_cols = ['rmssd_change', 'bbi_change', 'steps_change', 'stress_change', 'duration_minutes', 'impact_score']
    categorical_cols = ['instance_type', 'name', 'time_of_day', 'day_of_week', 'pre_stress_state', 'sentiment', 'reported_impact']

    features_numeric = results_df[feature_cols].fillna(0)
    features_categorical = pd.get_dummies(results_df[categorical_cols], drop_first=True)
    X = pd.concat([features_numeric, features_categorical], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    results_df['cluster'] = cluster_labels

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", s=50)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("PCA Projection of Clusters")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    st.pyplot(fig)

    st.write("### Cluster Profiles")
    cluster_profiles = results_df.groupby('cluster').agg({
        'stress_reduction': lambda x: (x == True).mean() * 100,
        'impact_score': 'mean',
        'duration_minutes': 'mean'
    }).reset_index()
    st.dataframe(cluster_profiles.style.format({"stress_reduction": "{:.1f}%", "impact_score": "{:.2f}", "duration_minutes": "{:.1f}"}))

    return results_df

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

    st.subheader(f"Physiological Impact Analysis for {username}")
    viz_tabs = st.tabs(["Overall Impact", "Pattern Analysis", "ML Analysis", "Detailed Results"])

    with viz_tabs[0]:
        st.write("### Overall Impact by Type")
        impact_by_type = plot_df.groupby('instance_type')['impact_score'].mean().reset_index()
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='instance_type', y='impact_score', data=impact_by_type, ax=ax1)
        ax1.set_title("Average Physiological Impact by Type")
        ax1.set_xlabel("Type")
        ax1.set_ylabel("Impact Score (higher is better)")
        st.pyplot(fig1)

        st.write("### Impact by Reported Sentiment")
        sentiment_df = plot_df[plot_df['sentiment'] != 'Unknown']
        if not sentiment_df.empty:
            impact_by_sentiment = sentiment_df.groupby('sentiment')['impact_score'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='sentiment', y='impact_score', data=impact_by_sentiment,
                        order=['Positive', 'Neutral', 'Negative'], ax=ax2)
            ax2.set_title("Average Impact by Reported Sentiment")
            ax2.set_xlabel("Sentiment")
            ax2.set_ylabel("Impact Score")
            st.pyplot(fig2)

        st.write("### Stress Reduction Success Rate by Event/Intervention Name")
        name_success = plot_df.groupby('name').agg({
            'stress_reduction': lambda x: (x == True).mean() * 100,
            'impact_score': 'mean'
        }).reset_index()
        if not name_success.empty:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            name_success = name_success.sort_values('stress_reduction', ascending=False)
            bars = sns.barplot(x='name', y='stress_reduction', data=name_success, ax=ax3)
            for i, bar in enumerate(bars.patches):
                success_rate = name_success['stress_reduction'].iloc[i]
                if success_rate >= 75:
                    bar.set_color('green')
                elif success_rate >= 50:
                    bar.set_color('lightgreen')
                elif success_rate >= 25:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            ax3.set_title("Stress Reduction Success Rate by Name")
            ax3.set_xlabel("Name (Event/Intervention)")
            ax3.set_ylabel("Success Rate (%)")
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
            ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
            st.pyplot(fig3)

    with viz_tabs[1]:
        st.write("### Contextual Factors Influencing Stress Reduction")
        st.markdown("""
        These visualizations identify patterns in situational factors that predict whether 
        specific interventions or events yield positive or negative outcomes in reducing stress.
        """)
        time_patterns = st.session_state.get('time_patterns')
        if time_patterns is not None and not time_patterns.empty:
            st.write("#### Success Rate by Time of Day")
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            sns.barplot(x='time_of_day', y='stress_reduction', hue='instance_type', 
                       data=time_patterns, ax=ax4)
            ax4.set_title("Success Rate by Time of Day")
            ax4.set_ylabel("Success Rate")
            st.pyplot(fig4)
        
        day_patterns = st.session_state.get('day_patterns')
        if day_patterns is not None and not day_patterns.empty:
            st.write("#### Success Rate by Day of Week")
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            sns.barplot(x='day_of_week', y='stress_reduction', hue='instance_type', 
                       data=day_patterns, ax=ax5)
            ax5.set_title("Success Rate by Day of Week")
            ax5.set_ylabel("Success Rate")
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig5)

    with viz_tabs[2]:
        results_df_ml = exploratory_ml_analysis(results_df, n_clusters=3)
        st.write("The ML analysis clusters events/interventions based on physiological and contextual features. Explore the PCA projection and cluster profiles above.")

    with viz_tabs[3]:
        st.write("### Detailed Analysis Results")
        display_df = results_df.copy()
        display_df['start_time'] = display_df['start_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['end_time'] = display_df['end_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_cols = [
            'username', 'instance_type', 'name', 'start_time', 'end_time',
            'duration_minutes', 'time_of_day', 'day_of_week', 'pre_stress_state',
            'sentiment', 'reported_impact', 'stress_reduction',
            'rmssd_change', 'bbi_change', 'stress_change', 'impact_score', 'notes'
        ]
        final_df = display_df[display_cols].copy()
        for col in ['duration_minutes', 'rmssd_change', 'bbi_change', 'stress_change', 'impact_score']:
            if col in final_df.columns:
                final_df[col] = final_df[col].round(2)
        st.dataframe(final_df)

# ------------------ 4. CLEAR OLD DATA (OPTIONAL) + UPLOAD ANNOTATIONS ------------------ #

def upload_annotations(username: str):
    """
    Allows uploading event/intervention annotations from a CSV/Excel file.
    Saves them exactly like manual entries.
    """
    st.subheader("Upload Annotations")
    st.markdown("Upload a CSV or Excel file with your events and interventions data.")
    st.info("All uploaded events and interventions will be associated with your account for feature extraction purposes.")

    # Optionally clear existing data
    clear_existing = st.checkbox("Clear all existing annotations before import", help="If checked, all your existing events and interventions will be deleted before importing new ones")

    uploaded_file = st.file_uploader("Choose an annotation file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Normalize column names
            df.columns = [' '.join(col.split()).strip() if isinstance(col, str) else col for col in df.columns]
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Expected columns mapping
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

            # Attempt to find columns automatically
            found_columns = {}
            for field, expected_col in expected_columns.items():
                for col in df.columns:
                    if isinstance(col, str) and expected_col.lower() in col.lower():
                        found_columns[field] = col
                        break

            # Let user adjust column mappings
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

            time_format = st.selectbox(
                "Time format in your data:",
                options=["12-hour (e.g., 1:30 PM)", "24-hour (e.g., 13:30)"],
                index=0
            )

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
                            # Note: Here we use the "username" column for deletion.
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

                # For each row, parse and insert into the correct table.
                for idx, row in df.iterrows():
                    try:
                        progress_bar.progress((idx + 1) / total_rows)
                        user_col = username
                        event_type_text = ""
                        if col_mapping['Event Type'] and not pd.isna(row[col_mapping['Event Type']]):
                            event_type_text = str(row[col_mapping['Event Type']]).strip().lower()

                        is_intervention = 'intervention' in event_type_text
                        table_name = 'interventions' if is_intervention else 'events'

                        # Get the name of the event/intervention
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

                            # Insert into database using record_event_in_database from sql_utils
                            # (Ensure that record_event_in_database is defined to insert into the correct table)
                            from sql_utils import record_event_in_database
                            record_event_in_database(
                                username,
                                start_ms,
                                end_ms,
                                activity_name,
                                var_name=table_name
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
    Debug function to check what's in the database tables.
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
    """Function to fix event/intervention categorization"""
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
                                cursor.execute(
                                    f"INSERT INTO interventions ({columns}) VALUES ({placeholders})",
                                    tuple(record_dict.values())
                                )
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
                                cursor.execute(
                                    f"INSERT INTO events ({columns}) VALUES ({placeholders})",
                                    tuple(record_dict.values())
                                )
                                cursor.execute("DELETE FROM interventions WHERE id = %s", (int_id,))
                                conn.commit()
                                st.success(f"Moved '{int_name}' to events.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error recategorizing: {e}")

# ------------------ 5. RUN THE FULL EXTRACTION APP ------------------ #

def run_stepper_extraction():
    """
    Main entry function for the Feature Extraction tool.
    This function is imported by main.py.
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
        hours_window = st.slider(
            "Hours to analyze before/after events:",
            min_value=1,
            max_value=24,
            value=2,
            help="Number of hours to analyze before and after each event or intervention"
        )

        st.markdown("""
        This feature analyzes the physiological impact of events and interventions by comparing measurements
        before and after each occurrence. It also uses exploratory ML (unsupervised clustering) to automatically
        extract physiologically relevant features and identify patterns that predict whether specific
        interventions or events yield positive or negative outcomes in reducing stress.
        """)

        if st.button("Extract Features"):
            with st.spinner("Analyzing physiological data..."):
                try:
                    results = analyze_physiological_impact(username, hours_window)
                    if results is not None and isinstance(results, pd.DataFrame) and not results.empty:
                        st.session_state['impact_results'] = results
                        visualize_analysis(results, username)
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Analysis as CSV",
                            data=csv,
                            file_name=f"{username}_feature_extraction.csv",
                            mime="text/csv"
                        )
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
                    st.error(f"Traceback: {traceback.format_exc()}")

    with tab2:
        upload_annotations(username)

    with tab3:
        debug_tables(username)
        recategorize_annotations(username)
