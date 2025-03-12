import datetime 
import time
import streamlit as st
import pandas as pd
import altair as alt
from sql_utils import *
from plot_utils import *

# Dictionary converting short var names to long var names
VAR_DICT = {
    "stress": "stressLevel",
    "daily_heart_rate": "beatsPerMinute",
    "respiration": "breathsPerMinute",
    "bbi": "bbi",
    "step": "steps"
}

OPTIONS_TO_VAR = {
    "Stress Level" : "stress",
    "Heart Rate" : "daily_heart_rate",
    "Respiration Rate": "respiration",
    "Beat-to-beat Interval": "bbi",
    "Steps Taken": "step",
}
VAR_TO_OPTIONS = {v:k for k,v in OPTIONS_TO_VAR.items()}

def questionaire(selected, var_name = "events"):

    # Define local offset to handle time input
    local_offset = -time.timezone * 1000
    
    with st.container():
        st.markdown("### Add New Entry")
        st.markdown("---")

        # 1. Activity Selection
        st.markdown("#### 1. Select Activity")
        options = fetch_past_options(st.session_state['user'], var_name=var_name)
        options.append("Other (please specify)")
        options = [o for o in options if o is not None]
        
        selected_option = st.selectbox(
            "Choose an activity:",
            options,
            label_visibility="collapsed"
        )
        
        if selected_option == "Other (please specify)":
            st.session_state['other_selected'] = True
            other_response = st.text_input("Specify new activity:")
        else:
            st.session_state['other_selected'] = False
            other_response = selected_option

        # 2. Time Selection
        st.markdown("#### 2. Time Range")
        try:
            selected_times = selected["selection"]["param_1"]["isoDate"]
            start_time = pd.to_datetime(selected_times[0] + local_offset, unit='ms')
            end_time = pd.to_datetime(selected_times[1] + local_offset, unit='ms')
            start_time = start_time.tz_localize(None)
            end_time = end_time.tz_localize(None)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Start**")
                start_date = st.date_input("Start date", value=start_time.date(), label_visibility="collapsed")
                start_time_input = st.time_input(
                    "Start time",
                    value=start_time.time(),
                    label_visibility="collapsed",
                    step=60
                )
            
            with col2:
                st.markdown("**End**")
                end_date = st.date_input("End date", value=end_time.date(), label_visibility="collapsed")
                end_time_input = st.time_input(
                    "End time",
                    value=end_time.time(),
                    label_visibility="collapsed",
                    step=60
                )

            # Combine date and time
            start_datetime = datetime.datetime.combine(start_date, start_time_input)
            end_datetime = datetime.datetime.combine(end_date, end_time_input)

            if end_datetime <= start_datetime:
                st.error("⚠️ End time must be after start time")
                return

            # Review and Submit
            with st.form("user_response_form", clear_on_submit=True):
                st.markdown("#### Review and Submit")
                st.markdown(f"""
                **Type:** {var_name.capitalize()}
                **Activity:** {other_response}
                **Time:** {start_datetime.strftime('%Y-%m-%d %H:%M')} to {end_datetime.strftime('%Y-%m-%d %H:%M')}
                """)
                
                submitted = st.form_submit_button("Submit Entry")
                if submitted:
                    try:
                        if st.session_state['other_selected']:
                            save_other_response(st.session_state['user'], other_response, var_name=var_name)
                        
                        # Convert to milliseconds timestamp for database
                        start_timestamp = int(start_datetime.timestamp() * 1000)
                        end_timestamp = int(end_datetime.timestamp() * 1000)
                        
                        record_event_in_database(
                            st.session_state['user'],
                            start_timestamp,
                            end_timestamp,
                            other_response,
                            var_name=var_name
                        )
                        st.success("✅ Entry recorded successfully!")
                        # Instead of immediate rerun, set a flag to redirect
                        st.session_state['submit_selection'] = False
                        st.session_state['show_annotations'] = True
                        time.sleep(3)  # Give user time to see success message
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saving entry: {str(e)}")
        except Exception as e:
            st.error("⚠️ Please select a time range on the plot first")
            return

def add_annotations(table_name, start_date, end_date, start_hour, end_hour):
    """
    Function that plots events/interventions/calendar events in the specified timeframe.
    """
    conn = get_rds_connection()

    try:
        # Different query handling for calendar events
        if table_name == 'calendar_events':
            query = f"""
                SELECT 
                    start_time,
                    end_time,
                    summary as {table_name}
                FROM {st.session_state.user}_{table_name}
            """
        else:
            # Query the specified table for the logged-in user's data
            query = f"SELECT * FROM {table_name} WHERE name = %s"
            
        # Try to execute the query
        try:
            if table_name == 'calendar_events':
                annotations_df = pd.read_sql_query(query, conn)
            else:
                annotations_df = pd.read_sql_query(query, conn, params=(st.session_state.user,))
                
            # Process the data if we have any
            if not annotations_df.empty:
                # Convert timestamps for events/interventions
                local_offset = -time.timezone * 1000
                
                if table_name == 'calendar_events':
                    # Calendar events are already in datetime format
                    annotations_df['start_time'] = pd.to_datetime(annotations_df['start_time'])
                    annotations_df['end_time'] = pd.to_datetime(annotations_df['end_time'])
                else:
                    # Convert timestamps for events/interventions
                    annotations_df['start_time'] = pd.to_datetime(annotations_df['start_time'] + local_offset, unit="ms")
                    annotations_df['end_time'] = pd.to_datetime(annotations_df['end_time'] + local_offset, unit="ms")

                # Filter the annotations
                filtered_df = annotations_df[
                    (annotations_df['start_time'].dt.date >= start_date.date()) &
                    (annotations_df['end_time'].dt.date <= end_date.date()) &
                    (annotations_df['start_time'].dt.hour >= start_hour) &
                    (annotations_df['end_time'].dt.hour <= end_hour)
                ]

                # Define color schemes for different tables
                color_scheme = {
                    'events': 'orangered',
                    'interventions': 'purpleblue',
                    'calendar_events': 'goldgreen'
                }

                chosen_color_scheme = color_scheme.get(table_name, 'category10')

                # Create the annotation chart
                if not filtered_df.empty:
                    annotation = alt.Chart(filtered_df).mark_rect(opacity=0.5).encode(
                        x='start_time:T',
                        x2='end_time:T',
                        color=alt.Color(f'{table_name}', legend=alt.Legend(title=table_name), scale=alt.Scale(scheme=chosen_color_scheme)),
                        tooltip=[
                            alt.Tooltip('start_time:T', title='Start Time', format=r"%c"),
                            alt.Tooltip('end_time:T', title='End Time', format=r"%c"),
                        ]
                    )
                    return annotation
                else:
                    # Return None if no data after filtering
                    return None
            else:
                # Return None if no data
                return None
                
        except (pd.errors.DatabaseError, pymysql.err.ProgrammingError, pymysql.err.OperationalError) as e:
            # Handle database errors (like table doesn't exist)
            if "doesn't exist" in str(e):
                if table_name == 'calendar_events':
                    st.info(f"No calendar events found. Please upload a calendar with events first.")
                else:
                    st.info(f"No {table_name} found. You can add {table_name} by selecting regions on the chart.")
            return None
            
    except Exception as e:
        # Handle any other exceptions
        st.error(f"Error loading {table_name}: {str(e)}")
        return None
    finally:
        # Always close the connection
        conn.close()

def diff_plot_util(selected_var_dfname):
    options = fetch_past_options(st.session_state['user'], var_name = "interventions")
    options = [o for o in options if o is not None]
    
    if not options:
        st.warning("No interventions found. Please add interventions before creating comparison plots.")
        return
        
    selected_option = st.selectbox("Please select an option:", options)
    
    # Ask the user for an integer input
    minutes_input_before = st.text_input("How many minutes before (X) should be:", "15")
    minutes_input_after = st.text_input("How many minutes after (X) should be:", "15")
    X_before, X_after = None, None 
    
    if minutes_input_before and minutes_input_after:
        # Validate if the input is an integer
        if minutes_input_before:
            try:
                # Convert the input to an integer
                X_before = int(minutes_input_before)
            except ValueError:
                st.error("Please enter a valid integer.")
        if minutes_input_after:
            try:
                # Convert the input to an integer
                X_after = int(minutes_input_after)
            except ValueError:
                st.error("Please enter a valid integer.")
    
    if X_before is not None and X_after is not None:
        try:
            result = get_instances(
                user=st.session_state.user,
                intervention=selected_option,
                mins_before=X_before,
                mins_after=X_after,
                var=selected_var_dfname,
                var_dict=VAR_DICT
            )
            
            if result is None:
                st.warning(f"No data available for {selected_option} with {selected_var_dfname}. Please remember to tag the interventions you want to compare.")
                return
            
            # Initialize session state for split/combined view if not found
            if "split_comparison_plots" not in st.session_state:
                st.session_state.split_comparison_plots = True

            if st.button("Switch Plot View"):
                st.session_state.split_comparison_plots = not st.session_state.split_comparison_plots
                
            instances_df, aggregate_df = result
            
            ComparisonPlotsManager(instances_df, aggregate_df, selected_var_dfname, VAR_TO_OPTIONS, X_before, X_after, split_comparison_plots=st.session_state.split_comparison_plots)
            
        except Exception as e:
            st.error(f"Error generating comparison plots: {str(e)}")
            st.info("This may be due to insufficient data for the selected intervention and variable.")

def visualization_page(annotation = False, diff_plot = False):
    st.title("Visualization Page")

    if 'submit_selection' not in st.session_state:
        st.session_state['submit_selection'] = False
    if 'other_selected' not in st.session_state:
        st.session_state['other_selected'] = False

    var_options = [k for k in OPTIONS_TO_VAR]

    selected_var = st.selectbox("Please select an option:", var_options)
    selected_var_dfname = OPTIONS_TO_VAR[selected_var]

    # Fetch data
    df = fetch_data(selected_var, OPTIONS_TO_VAR, st.session_state.user)

    # Check if DataFrame is empty
    if df.empty:
        # Check if there's an error message
        if hasattr(df, 'attrs') and 'error_message' in df.attrs:
            st.warning(df.attrs['error_message'])
        else:
            st.info("No data available for visualization. Please upload or collect data first.")
        return  # Exit the function early
    
    df['isoDate'] = pd.to_datetime(df['timestamp_cleaned'])

    # Date and time range selection
    start_date = df['isoDate'].min()
    end_date = df['isoDate'].max()

    start_date = st.sidebar.date_input('Start Date', value=start_date)
    end_date = st.sidebar.date_input('End Date', value=end_date)
    start_hour, end_hour = st.sidebar.slider("Select Hour Range", 0, 23, (0, 23))

    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)

    filtered_df = df[
        (df['isoDate'].dt.date >= start_date.date()) &
        (df['isoDate'].dt.date <= end_date.date()) &
        (df['isoDate'].dt.hour >= start_hour) &
        (df['isoDate'].dt.hour <= end_hour)
    ]
    
    # Check if filtered DataFrame is empty
    if filtered_df.empty:
        st.warning("No data available for the selected time range. Please adjust your filters.")
        return  # Exit the function early

    chart = get_plot(selected_var, filtered_df)

    if annotation:
        show_events = st.sidebar.checkbox('Show Events')
        show_interventions = st.sidebar.checkbox('Show Interventions')
        show_calendar = st.sidebar.checkbox('Show Calendar Events')

        # Start with the base chart
        final_chart = chart

        if show_events:
            event_annotation = add_annotations('events', start_date, end_date, start_hour, end_hour)
            if event_annotation is not None:
                final_chart = alt.layer(final_chart, event_annotation).resolve_scale(color='independent')

        if show_interventions:
            interventions_annotation = add_annotations('interventions', start_date, end_date, start_hour, end_hour)
            if interventions_annotation is not None:
                final_chart = alt.layer(final_chart, interventions_annotation).resolve_scale(color='independent')

        if show_calendar:
            calendar_annotation = add_annotations('calendar_events', start_date, end_date, start_hour, end_hour)
            if calendar_annotation is not None:
                final_chart = alt.layer(final_chart, calendar_annotation).resolve_scale(color='independent')

        # Add padding to the final layered chart
        final_chart = final_chart.properties(
            padding={'left': 60, 'top': 20, 'right': 20, 'bottom': 30}
        )

        st.altair_chart(final_chart, use_container_width=True)

    elif diff_plot:
        diff_plot_util(selected_var_dfname)

    else:
        selection = alt.selection_interval(encodings=['x'])
        selected = None 
        chart = chart.add_selection(selection)
        selected = st.altair_chart(chart, use_container_width=True, on_select="rerun")

        if st.button("Enter Events"):
            st.session_state['submit_selection'] = True

        if st.session_state['submit_selection']:
            st.write("You've selected a region! Please answer some questions.")
            choice = st.radio("Select an option:", ('Events', 'Interventions'))
            
            # Check if there's a valid selection
            if selected is not None and "selection" in selected and "param_1" in selected["selection"]:
                if choice == 'Events':
                    questionaire(selected, var_name="events")
                elif choice == 'Interventions':
                    questionaire(selected, var_name="interventions")
            else:
                st.info("Please select a time range by clicking and dragging on the plot")

if __name__ == '__main__':
    visualization_page()