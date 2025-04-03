import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
from datetime import datetime, timedelta
import re
from sql_utils import get_rds_connection
import streamlit as st
import altair as alt  # Add this import

# ML imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap


# Import VAR_DICT from your visualization module (adjust the path as needed)
from visualization_page import VAR_DICT  

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
                        df[col] = pd.to_datetime(df[col], unit='ms')
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

# ------------------ EXPLORATORY ML & VISUALIZATIONS ------------------ #
def exploratory_ml_analysis(results_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    st.subheader("Exploratory ML Analysis")
    if results_df.empty:
        st.warning("No data to analyze.")
        return results_df
    
    # Prepare features and scale them
    feature_cols = ['rmssd_change', 'bbi_change', 'steps_change', 'stress_change', 'duration_minutes', 'impact_score']
    categorical_cols = ['instance_type', 'name', 'time_of_day', 'day_of_week', 'pre_stress_state', 'sentiment', 'reported_impact']
    features_numeric = results_df[feature_cols].fillna(0)
    features_categorical = pd.get_dummies(results_df[categorical_cols], drop_first=True)
    X = pd.concat([features_numeric, features_categorical], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    results_df['cluster'] = cluster_labels
    
    # Determine TSNE perplexity dynamically based on the sample size.
    n_samples = X_scaled.shape[0]
    tsne_perplexity = min(30, max(5, n_samples - 1))
    
    # Build projection DataFrames with additional temporal info:
    # Date and Time are extracted from the start_time column.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        'Dim1': X_pca[:, 0],
        'Dim2': X_pca[:, 1],
        'Cluster': cluster_labels,
        'Name': results_df['name'],
        'Type': results_df['instance_type'],
        'Date': results_df['start_time'].dt.strftime('%Y-%m-%d'),
        'Time': results_df['start_time'].dt.strftime('%H:%M'),
        'Day': results_df['day_of_week'],
        'TimeOfDay': results_df['time_of_day']
    })
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_df = pd.DataFrame({
        'Dim1': X_tsne[:, 0],
        'Dim2': X_tsne[:, 1],
        'Cluster': cluster_labels,
        'Name': results_df['name'],
        'Type': results_df['instance_type'],
        'Date': results_df['start_time'].dt.strftime('%Y-%m-%d'),
        'Time': results_df['start_time'].dt.strftime('%H:%M'),
        'Day': results_df['day_of_week'],
        'TimeOfDay': results_df['time_of_day']
    })
    

    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X_scaled)
    umap_df = pd.DataFrame({
        'Dim1': X_umap[:, 0],
        'Dim2': X_umap[:, 1],
        'Cluster': cluster_labels,
        'Name': results_df['name'],
        'Type': results_df['instance_type'],
        'Date': results_df['start_time'].dt.strftime('%Y-%m-%d'),
        'Time': results_df['start_time'].dt.strftime('%H:%M'),
        'Day': results_df['day_of_week'],
        'TimeOfDay': results_df['time_of_day']
    })
    
    # Create sub-tabs for PCA, t-SNE, and UMAP projections
    dim_tabs = st.tabs(["PCA Projection", "t-SNE Projection", "UMAP Projection"])
    
    tooltip_fields = ['Name:N', 'Type:N', 'Date:T', 'Time:N', 'Day:N', 'TimeOfDay:N']
    
    with dim_tabs[0]:
        pca_chart = alt.Chart(pca_df).mark_circle(size=60).encode(
            x=alt.X('Dim1:Q', title='PCA Component 1'),
            y=alt.Y('Dim2:Q', title='PCA Component 2'),
            color=alt.Color('Cluster:N', scale=alt.Scale(scheme='viridis')),
            tooltip=tooltip_fields
        ).properties(
            width=700,
            height=500,
            title='PCA Projection of Events/Interventions'
        ).interactive()
        st.altair_chart(pca_chart, use_container_width=True)
    
    with dim_tabs[1]:
        tsne_chart = alt.Chart(tsne_df).mark_circle(size=60).encode(
            x=alt.X('Dim1:Q', title='t-SNE Dim 1'),
            y=alt.Y('Dim2:Q', title='t-SNE Dim 2'),
            color=alt.Color('Cluster:N', scale=alt.Scale(scheme='viridis')),
            tooltip=tooltip_fields
        ).properties(
            width=700,
            height=500,
            title='t-SNE Projection of Events/Interventions'
        ).interactive()
        st.altair_chart(tsne_chart, use_container_width=True)
    
    with dim_tabs[2]:
        umap_chart = alt.Chart(umap_df).mark_circle(size=60).encode(
            x=alt.X('Dim1:Q', title='UMAP Dim 1'),
            y=alt.Y('Dim2:Q', title='UMAP Dim 2'),
            color=alt.Color('Cluster:N', scale=alt.Scale(scheme='viridis')),
            tooltip=tooltip_fields
        ).properties(
            width=700,
            height=500,
            title='UMAP Projection of Events/Interventions'
        ).interactive()
        st.altair_chart(umap_chart, use_container_width=True)
    
    # Brief explanation of components and dimensions
    st.markdown("""
    ### Explanation of Components and Dimensions
    
    **PCA Components:**  
    PCA creates new variables (components) that are linear combinations of the original features.  
    - **Component 1:** Captures the maximum variance in the data (i.e., the strongest pattern).  
    - **Component 2:** Captures the next most significant independent pattern.  
    
    **t-SNE Dimensions:**  
    t-SNE transforms high-dimensional data into two dimensions while preserving local neighborhood structure.  
    - These dimensions are latent features optimized for visualization rather than direct interpretation.
    
    **UMAP Dimensions:**  
    UMAP also produces a two-dimensional embedding that preserves both local and some global data structure.  
    - Like t-SNE, its dimensions are latent and are useful for uncovering the manifold structure of the data.
    
    Hovering over any point in the projections will reveal details such as the event/intervention name, type, date, time, day of the week, and time of day.
    """)
    
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
    st.subheader(f"Physiological Impact Analysis for User: {username}")
    viz_tabs = st.tabs(["Overall Impact", "Pattern Analysis", "ML Analysis", "Detailed Results"])
    
    with viz_tabs[0]:
        st.write("### Overall Impact by Type")
        impact_by_type = plot_df.groupby('instance_type')['impact_score'].mean().reset_index()
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        impact_colors = ['green' if score > 0 else 'red' for score in impact_by_type['impact_score']]
        bars = ax1.bar(impact_by_type['instance_type'], impact_by_type['impact_score'], color=impact_colors)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        for i, p in enumerate(bars):
            height = p.get_height()
            y_pos = height - 5 if height < 0 else height + 5
            text_color = 'white' if height < 0 else 'black'
            ax1.annotate(f'{height:.1f}', (p.get_x() + p.get_width()/2, y_pos),
                         ha='center', va='center', fontsize=12, color=text_color, fontweight='bold')
        ax1.set_title("Average Physiological Impact by Type")
        ax1.set_xlabel("Type")
        ax1.set_ylabel("Impact Score (higher is better)")
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
        legend_elements = [plt.Rectangle((0,0),1,1,color='green', label='Positive (Beneficial)'),
                           plt.Rectangle((0,0),1,1,color='red', label='Negative (Detrimental)')]
        ax1.legend(handles=legend_elements, loc='best')
        st.pyplot(fig1)
        st.markdown("""
        **Impact Score Key:**
        
        Impact Score is computed automatically using standardized changes from all physiological metrics.
        For each metric (like stress_change, rmssd_change, bbi_change, etc.), the percent change is converted to a z‑score
        (i.e. (value - mean) / standard deviation). For metrics where a decrease is beneficial (e.g. stress_change), the z‑score
        is multiplied by –1. The final Impact Score is the average of these standardized values.
        This approach allows the data itself to determine the relative importance of each metric without manually setting weights.
        """)
        
        st.write("### Impact by Reported Sentiment")
        sentiment_df = plot_df[plot_df['sentiment'] != 'Unknown']
        if not sentiment_df.empty:
            impact_by_sentiment = sentiment_df.groupby('sentiment')['impact_score'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sentiment_colors = ['green' if score > 0 else 'red' for score in impact_by_sentiment['impact_score']]
            sentiment_order = ['Positive', 'Neutral', 'Negative']
            ordered_sentiment = pd.concat([impact_by_sentiment[impact_by_sentiment['sentiment'] == s] for s in sentiment_order if s in impact_by_sentiment['sentiment'].values])
            if not ordered_sentiment.empty:
                bars = ax2.bar(ordered_sentiment['sentiment'], ordered_sentiment['impact_score'],
                               color=[sentiment_colors[impact_by_sentiment['sentiment'].tolist().index(s)] for s in ordered_sentiment['sentiment']])
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                for i, p in enumerate(bars):
                    height = p.get_height()
                    y_pos = height - 5 if height < 0 else height + 5
                    text_color = 'white' if height < 0 else 'black'
                    ax2.annotate(f'{height:.1f}', (p.get_x() + p.get_width()/2, y_pos),
                                 ha='center', va='center', fontsize=12, color=text_color, fontweight='bold')
                ax2.set_title("Average Impact by Reported Sentiment")
                ax2.set_xlabel("Sentiment")
                ax2.set_ylabel("Impact Score (higher is better)")
                legend_elements = [plt.Rectangle((0,0),1,1,color='green', label='Positive (Beneficial)'),
                                   plt.Rectangle((0,0),1,1,color='red', label='Negative (Detrimental)')]
                ax2.legend(handles=legend_elements, loc='best')
                plt.tight_layout()
                st.pyplot(fig2)
                if 'Positive' in ordered_sentiment['sentiment'].values:
                    pos_impact = ordered_sentiment[ordered_sentiment['sentiment'] == 'Positive']['impact_score'].iloc[0]
                    if pos_impact < 0:
                        st.markdown("""
                        > **Note:** Some events labeled as positive show negative physiological effects.
                        """)
    
    with viz_tabs[1]:
        # Standardized Metrics Visualizations
        st.write("### Which events/interventions are most effective at reducing stress")
        name_stress_change = plot_df.groupby('name')['stress_change'].mean().reset_index()
        mean_stress = name_stress_change['stress_change'].mean()
        std_stress = name_stress_change['stress_change'].std() if name_stress_change['stress_change'].std() != 0 else 1e-6
        name_stress_change['std_stress'] = -1 * (name_stress_change['stress_change'] - mean_stress) / std_stress
        name_stress_change['std_stress_pct'] = name_stress_change['std_stress'] * 100
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        name_stress_change = name_stress_change.sort_values('std_stress')
        stress_colors = ['green' if val > 0 else 'red' for val in name_stress_change['std_stress_pct']]
        bars = ax3.bar(name_stress_change['name'], name_stress_change['std_stress_pct'], color=stress_colors)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        for i, p in enumerate(bars):
            pct_val = name_stress_change['std_stress_pct'].iloc[i]
            y_pos = pct_val - 5 if pct_val < 0 else pct_val + 5
            text_color = 'white' if pct_val < 0 else 'black'
            ax3.annotate(f'{pct_val:.1f}%', (p.get_x() + p.get_width()/2, y_pos),
                         ha='center', va='center', fontsize=11, color=text_color, fontweight='bold')
        ax3.set_title("Standardized Stress Reduction by Event/Intervention Name")
        ax3.set_xlabel("Event/Intervention")
        ax3.set_ylabel("Standardized Stress Reduction (%; higher is better)")
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.write("### HRV Change by Event/Intervention Name")
        name_hrv_change = plot_df.groupby('name')['rmssd_change'].mean().reset_index()
        mean_hrv = name_hrv_change['rmssd_change'].mean()
        std_hrv = name_hrv_change['rmssd_change'].std() if name_hrv_change['rmssd_change'].std() != 0 else 1e-6
        name_hrv_change['std_hrv'] = (name_hrv_change['rmssd_change'] - mean_hrv) / std_hrv
        name_hrv_change['std_hrv_pct'] = name_hrv_change['std_hrv'] * 100
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        name_hrv_change = name_hrv_change.sort_values('std_hrv', ascending=False)
        hrv_colors = ['green' if val > 0 else 'red' for val in name_hrv_change['std_hrv_pct']]
        bars = ax4.bar(name_hrv_change['name'], name_hrv_change['std_hrv_pct'], color=hrv_colors)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        for i, p in enumerate(bars):
            pct_val = name_hrv_change['std_hrv_pct'].iloc[i]
            y_pos = pct_val - 5 if pct_val < 0 else pct_val + 5
            text_color = 'white' if pct_val < 0 else 'black'
            ax4.annotate(f'{pct_val:.1f}%', (p.get_x() + p.get_width()/2, y_pos),
                         ha='center', va='center', fontsize=11, color=text_color, fontweight='bold')
        ax4.set_title("Standardized HRV Change by Event/Intervention Name")
        ax4.set_xlabel("Event/Intervention")
        ax4.set_ylabel("Standardized HRV Change (%; higher is better)")
        plt.tight_layout()
        st.pyplot(fig4)
        
        st.write("### BBI Change by Event/Intervention Name")
        name_bbi_change = plot_df.groupby('name')['bbi_change'].mean().reset_index()
        mean_bbi = name_bbi_change['bbi_change'].mean()
        std_bbi = name_bbi_change['bbi_change'].std() if name_bbi_change['bbi_change'].std() != 0 else 1e-6
        name_bbi_change['std_bbi'] = (name_bbi_change['bbi_change'] - mean_bbi) / std_bbi
        name_bbi_change['std_bbi_pct'] = name_bbi_change['std_bbi'] * 100
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        name_bbi_change = name_bbi_change.sort_values('std_bbi', ascending=False)
        bbi_colors = ['green' if val > 0 else 'red' for val in name_bbi_change['std_bbi_pct']]
        bars = ax5.bar(name_bbi_change['name'], name_bbi_change['std_bbi_pct'], color=bbi_colors)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        for i, p in enumerate(bars):
            pct_val = name_bbi_change['std_bbi_pct'].iloc[i]
            y_pos = pct_val - 5 if pct_val < 0 else pct_val + 5
            text_color = 'white' if pct_val < 0 else 'black'
            ax5.annotate(f'{pct_val:.1f}%', (p.get_x() + p.get_width()/2, y_pos),
                         ha='center', va='center', fontsize=11, color=text_color, fontweight='bold')
        ax5.set_title("Standardized BBI Change by Event/Intervention Name")
        ax5.set_xlabel("Event/Intervention")
        ax5.set_ylabel("Standardized BBI Change (%; higher is better)")
        plt.tight_layout()
        st.pyplot(fig5)
        
        
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

        st.write("#### Heatmap: Average Standardized Stress Reduction by Day of Week and Time of Day")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
        sns.heatmap(temporal_pivot, annot=True, cmap='viridis', fmt=".2f", ax=ax_heat)
        ax_heat.set_title("Standardized Stress Reduction by Day and Time")
        ax_heat.set_xlabel("Time of Day")
        ax_heat.set_ylabel("Day of Week")
        st.pyplot(fig_heat)

    
    with viz_tabs[2]:
        results_df_ml = exploratory_ml_analysis(results_df, n_clusters=3)
        st.write("The ML analysis clusters events/interventions based on physiological and contextual features. Explore the PCA, t-SNE, and UMAP projections above.")
    
    with viz_tabs[3]:
        st.write("### Detailed Analysis Results")
        display_df = results_df.copy()
        display_df['start_time'] = display_df['start_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['end_time'] = display_df['end_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_cols = ['username', 'instance_type', 'name', 'start_time', 'end_time',
                        'duration_minutes', 'time_of_day', 'day_of_week', 'pre_stress_state',
                        'sentiment', 'reported_impact', 'stress_reduction',
                        'rmssd_change', 'bbi_change', 'stress_change', 'impact_score', 'notes']
        final_df = display_df[display_cols].copy()
        for col in ['duration_minutes', 'rmssd_change', 'bbi_change', 'stress_change', 'impact_score']:
            if col in final_df.columns:
                final_df[col] = final_df[col].round(2)
        st.dataframe(final_df)

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
        hours_window = st.slider("Hours to analyze before/after events:", min_value=1, max_value=24, value=2,
                                 help="Number of hours to analyze before and after each event or intervention")
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
                    if results is not None and not results.empty:
                        st.session_state['impact_results'] = results
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
    
    with tab2:
        upload_annotations(username)
    
    with tab3:
        recategorize_annotations(username)
