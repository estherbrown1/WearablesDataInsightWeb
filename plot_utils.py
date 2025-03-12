import streamlit as st
import pandas as pd
import altair as alt
from dataclasses import dataclass, field
from vega_datasets import data
from sklearn.linear_model import LinearRegression
import numpy as np

@dataclass
class ComparisonPlotsManager:
    """
    """
    # Attributes of the class
    instances_df: pd.DataFrame
    aggregate_df: pd.DataFrame
    var: str
    var_to_label: dict
    mins_before: int
    mins_after: int
    split_comparison_plots: bool = False
    max_instances: int = 5
    max_duration: float = 0.
    var_label: str = ""
    differences_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        # Only display valid data
        self.instances_df = self.instances_df[self.instances_df[self.var] > 0]
        # Don't plot anything if no data are found
        if self.instances_df.empty:
            st.write(f"There are no valid data for {self.var_to_label[self.var].lower()} for any intervention.")
            return
        # Initialize neat variable name for labelling
        self.var_label = self.var.title().replace("_"," ") if len(self.var) > 3 else self.var.upper()
        # Get duration of longest intervention
        self.max_duration = (self.instances_df["duration"].dt.total_seconds() / 60).max()
        # Update segments to prevent interpolation of far points
        self.instances_df = update_time_segments(self.instances_df)
        # Add effect column to dataframes
        self.update_effect()
        # Plot line plots with each instance trajectory
        self.plot_trajectories()
        # Plot bar plot with aggregated data
        self.plot_aggregate()

    @staticmethod
    def combine_layers(layers:list[alt.Chart], title:str="", include_props:bool=False, resolve:str="independent") -> alt.Chart:
        """Helper function to combine altair chart layers."""
        plot = alt.layer(
            *layers
        ).resolve_scale(
            color=resolve,
            stroke=resolve,
        )

        if include_props:
            plot = plot.properties(
                width=800,
                height=400,
                title=alt.Title(
                    title, 
                    anchor="middle"
                )
            )
        return plot
                

    @staticmethod
    def update_time_segments(df:pd.DataFrame, time_gap_threshold:str="5min"):
        """
        Helper function to split non-continuous data points according to the time_gap_threshold.
        """
        # Identify unique segments to separate lines in the plot
        copy_df = df.copy()
        if copy_df["timestamp_cleaned"].isna().sum() > 0:
            copy_df['timestamp_cleaned'] = pd.to_datetime(copy_df['unix_timestamp_cleaned'], unit='ms')
        if "isoDate" not in copy_df.columns:
            copy_df["isoDate"] = pd.to_datetime(copy_df['timestamp_cleaned'])
        copy_df = copy_df.sort_values(by='isoDate')
        copy_df["time_diff"] = copy_df["isoDate"].diff()
        threshold = pd.Timedelta(time_gap_threshold)
        copy_df['segment'] = (copy_df['time_diff'] > threshold).cumsum()
        copy_df = copy_df.drop(columns=["time_diff"])
        return copy_df
    
    def update_effect(self) -> None:
        """
        Method that updates the 'effect' column in instances_df and aggregate_df.
        'Positive' if average conditions improve after the intervention, 'Negative' otherwise
        """
        self.differences_df = pd.DataFrame(index=self.aggregate_df.index, columns=["effect"])
        self.differences_df["mean"] = (self.aggregate_df[((self.var,"mean"), "after")] - self.aggregate_df[((self.var,"mean"), "before")]) / self.aggregate_df[((self.var,"mean"), "before")]
        self.differences_df = self.differences_df.reset_index() # reset index to plot index as x-axis

        # Determine if effect was positive based on variable type
        if self.var in ["stress", "daily_heart_rate", "respiration"]:
            # For stress and heart rate, decrease is positive
            self.differences_df.loc[self.differences_df["mean"] < 0,"effect"] = "Positive"
            self.differences_df.loc[self.differences_df["mean"] >= 0,"effect"] = "Negative"
        elif self.var == "bbi":
            # For bbi, increase is positive
            self.differences_df.loc[self.differences_df["mean"] > 0,"effect"] = "Positive"
            self.differences_df.loc[self.differences_df["mean"] <= 0,"effect"] = "Negative"
        else:
            self.differences_df["effect"] = "Neutral"

        # Update instances_df with effect
        for instance in self.differences_df["instance"].unique():
            self.instances_df.loc[self.instances_df["instance"] == instance, "effect"] = self.differences_df[self.differences_df["instance"] == instance]["effect"].iloc[0]

    def plot_trajectories(self) -> None:
        """
        Method to plot the trajectory of instances as lines.
        """
        # Get minimum and maximum of the variable
        var_min, var_max = self.instances_df[self.var].min(),  self.instances_df[self.var].max()

        # Create custom x axis and y scale
        x_axis = self.create_trajectory_x_axis()
        y_scale = alt.Scale(domain=[var_min, var_max], padding=0.15)

        # Add labels
        labels_layers = self.create_trajectory_annotations()

        # Initialize list of Altair layers
        all_line_layers, all_sleep_layers, all_regression_layers, all_event_layers = [], [], [], []
        # Initialize list of subplots 
        all_subplots = []

        # Initialize inclusion of title
        include_props = True

        # Iterate through possible effects
        for effect, effect_color in zip(["Positive", "Negative"],["blues", "reds"]):
            # Filter based on effect of the intervention
            effects_df = self.instances_df[self.instances_df["effect"] == effect]
            if effects_df.empty: pass

            # Define the conditional color logic for sleep data
            color_condition = alt.condition(
                alt.datum.sleep_label == "No Sleep Data",  
                alt.value("gray"),  
                alt.Color("instance:N", scale=alt.Scale(scheme=effect_color), legend=None)
            )
            
            # Create separate plots
            if self.split_comparison_plots:
                dfs = [effects_df[effects_df["instance"] == instance] for instance in effects_df["instance"].unique()]
                dfs = dfs[:self.max_instances]
            # Only a single plot
            else:
                dfs = [effects_df]

            # Iterate through dataframes and plot each one
            for instance_df in dfs:
                instance_name = instance_df['instance'].iloc[0]
                instances_to_process = [instance_name] if self.split_comparison_plots else instance_df["instance"].unique()
                # Aggregate data for sleep reporting
                sleep_df = instance_df.sort_values("mins").groupby("instance").aggregate({
                    "mins": "last", 
                    self.var: "last", 
                    "sleep_score": "last", 
                    "sleep_label": "last",
                    "effect": "first"  # Include effect in aggregation
                }).reset_index()
                sleep_df["dummy_outline"] = 100
                # Plot the various components of the plot
                instance_line_layers = self.create_trajectory_lines(instance_df, effect, effect_color, x_axis, y_scale)
                instance_sleep_layers = self.create_trajectory_sleep_info(sleep_df, color_condition)
                instance_regression_layers = self.create_trajectory_regression(instance_df, instances_to_process, x_axis, y_scale, effect_color)
                instance_event_layer = self.create_trajectory_events(instance_df)
                # Immediately display the plot if choosing split view
                if self.split_comparison_plots:
                    subplot_title = f"{self.var_label} Before, During, and After Intervention - {instance_name}"
                    subplot = self.combine_layers([*labels_layers, *instance_line_layers, *instance_sleep_layers, *instance_regression_layers, instance_event_layer], title=f"{self.var_label} Before, During, and After Interventions", include_props=include_props)
                    st.altair_chart(subplot, use_container_width=True)
                    include_props = False
                # Combine into a single list of all layers if choosing combined view
                else:
                    all_line_layers.extend(instance_line_layers)
                    all_sleep_layers.extend(instance_sleep_layers)
                    all_regression_layers.extend(instance_regression_layers)
                    all_event_layers.append(instance_event_layer)

        if not self.split_comparison_plots:
            combined_event_layer = self.combine_layers(all_event_layers, resolve="shared")
            plot = self.combine_layers([*labels_layers, *all_line_layers, *all_sleep_layers, *all_regression_layers, combined_event_layer], title=f"{self.var_label} Before, During, and After Interventions", include_props=True)
            st.altair_chart(plot, use_container_width=True)

    def create_trajectory_x_axis(self) -> alt.X:
        """
        Creates and returns a custom Altair x-axis for the trajectory plot.
        """
        custom_tick_vals = [i for i in range(0,self.mins_before,10)]
        custom_tick_vals.extend([self.mins_before+i for i in range(0, int(self.max_duration), 10)])
        custom_tick_vals.extend([self.mins_before+int(self.max_duration)+i for i in range(0,self.mins_after+11,10)])
        print(custom_tick_vals)
        x_axis = alt.X("mins:Q", 
            axis=alt.Axis(
                title="Minutes",
                values=custom_tick_vals,
                format="d",
                labelExpr=f"""
                    datum.value < {self.mins_before} ? -(datum.value - {self.mins_before})
                    : datum.value >= {self.mins_before+int(self.max_duration)} ? 
                      datum.value - {self.mins_before+int(self.max_duration)}
                    : datum.value - {self.mins_before}
                """
            )
        )
        return x_axis
    
    def create_trajectory_annotations(self, y_offset=-5) -> list[alt.Chart, alt.Chart, alt.Chart]:
        """
        Creates and returns Altair chart layers with text annotations of the chart.
        """
        # Add 'before', 'during', and 'after' as annotations on the plot
        labels_df = pd.DataFrame({
            "text": ["Before", "During", "After"],
            "x": [self.mins_before / 2, 
                  self.mins_before + (self.max_duration / 2), 
                  self.mins_before + self.max_duration + (self.mins_after / 2)]
        })
        section_labels = alt.Chart(labels_df).mark_text(
            align="center",
            baseline="bottom",
            fontSize=14,
            color="red"
        ).encode(
            x="x:Q",
            y=alt.value(y_offset),
            text=alt.Text("text:N"),
            tooltip=alt.value(None),
        )
        # Plot the start and end of the intervention/event
        vbar_marks = alt.Chart(pd.DataFrame({
            "x": [self.mins_before, self.mins_before + self.max_duration]
        })).mark_rule(color="red").encode(x="x:Q",tooltip=alt.value(None))
        return [section_labels, vbar_marks]
    
    def create_trajectory_lines(self, df:pd.DataFrame, effect:str, effect_color:str, 
                                x_axis:alt.X, y_scale:alt.Scale) -> tuple[alt.Chart, alt.Chart]:
        """
        Creates and returns Altair chart layers for lines representing trajectories of instances.
        """
        lines_before_after = alt.Chart(df[
            (df["status"].isin(["before", "after"])) & 
            df["event_name"].isna() & 
            df["calendar_name"].isna()
        ]).mark_line(
            opacity=0.75
        ).encode(
            x=x_axis,
            y=alt.Y(f"{self.var}:Q", axis=alt.Axis(title=self.var_label), scale=y_scale),
            color=alt.Color('instance:N', scale=alt.Scale(scheme=effect_color), legend=alt.Legend(title=f'Instance ({effect} Effect)')),
            detail=alt.Detail(['status:N', 'segment:N', 'instance:N']),
            tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"{self.var}:Q", title=self.var_label.title()),
                alt.Tooltip("instance:N", title="Instance")
            ]
        )
        lines_during = alt.Chart(df[df["status"] == "during"]).mark_line(
                opacity=0.75
            ).encode(
                x=x_axis,
                y=alt.Y(f"{self.var}:Q", scale=y_scale),
                color=alt.Color('instance:N', scale=alt.Scale(scheme=effect_color), legend=None),
                detail=alt.Detail('instance:N'),
                tooltip=[
                    alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                    alt.Tooltip(f"{self.var}:Q", title=self.var_label.title()),
                    alt.Tooltip("instance:N", title="Instance"),
                ]
            )
        return lines_before_after, lines_during
    
    def create_trajectory_sleep_info(self, sleep_df:pd.DataFrame, color_condition:alt.condition, inner_radius=6, outer_radius=12, x_padding:float=1.05) -> tuple[alt.Chart, alt.Chart, alt.Chart]:
        """
        Creates and returns Altair chart layers for sleep information
        """
        shared_x_max = 0 if self.split_comparison_plots else self.instances_df["mins"].max()
        outline = alt.Chart(sleep_df).mark_arc(innerRadius=inner_radius, outerRadius=outer_radius, strokeWidth=2, fillOpacity=0).encode(
            theta = alt.Theta("dummy_outline:Q", stack=False, scale=alt.Scale(domain=[0,100])),
            x = alt.X("modified_x:Q"),
            y = alt.Y(f"{self.var}:Q"),
            color=color_condition,
            stroke=color_condition,
            tooltip=[
                alt.Tooltip("instance:N", title="Instance"),
                alt.Tooltip("sleep_label:N", title="Sleep Quality"),
                alt.Tooltip("sleep_score:Q", title="Sleep Score"),
            ]
        ).transform_calculate(
            "modified_x",f"max(datum.mins * {x_padding}, {shared_x_max} * {x_padding})"
        )
        arcs = alt.Chart(sleep_df).mark_arc(innerRadius=inner_radius, outerRadius=outer_radius).encode(
            theta=alt.Theta("sleep_score:Q", stack=False, scale=alt.Scale(domain=[0,100])),
            x=alt.X("modified_x:Q"),
            y=alt.Y(f"{self.var}:Q"),
            color=color_condition,
            tooltip=[
                alt.Tooltip("instance:N", title="Instance"),
                alt.Tooltip("sleep_label:N", title="Sleep Quality"),
                alt.Tooltip("sleep_score:Q", title="Sleep Score"),
            ]
        ).transform_calculate(
            "modified_x",f"max(datum.mins * {x_padding}, {shared_x_max} * {x_padding})"
        )
        sleep_labels = alt.Chart(sleep_df).mark_text(yOffset=-inner_radius-outer_radius).encode(
            text=alt.Text("sleep_label:N"),
            x=alt.X("modified_x:Q"),
            y=alt.Y(f"{self.var}:Q"),
            color=color_condition,
            tooltip=alt.value(None),
        ).transform_calculate(
            "modified_x",f"max(datum.mins * {x_padding}, {shared_x_max} * {x_padding})"
        )
        return outline, arcs, sleep_labels
    
    def create_trajectory_regression(self, instance_df:pd.DataFrame, 
                                     instances_to_process:list[str], 
                                     x_axis:alt.X, y_scale:alt.Scale, color_scheme:str,
                                     regression_method:str="simple") -> list[alt.Chart]:
        """
        Method that creates regression trend lines using simple linear regression.
        """
        # Initialize list of regression layers
        regression_layers = []
        # Create regression lines for each instance and period separately
        for instance in instances_to_process:
            # Process each period separately
            for period in ['before', 'during', 'after']:
                # Get data for this instance and period
                period_data = instance_df[(instance_df['instance'] == instance) & 
                                        (instance_df['status'] == period)]
                
                # Only proceed if there are enough points for linear regression
                if len(period_data) > 1:  # Need at least 2 points for linear regression
                    try:
                        # Sort data by 'mins'
                        period_data = period_data.sort_values("mins")
                        x_vals = period_data["mins"].values
                        y_vals = period_data[self.var].values
                        
                        # Simple linear regression
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        x_reshaped = x_vals.reshape(-1, 1)
                        model.fit(x_reshaped, y_vals)
                        
                        # Generate predictions
                        x_hat = np.linspace(x_vals.min(), x_vals.max(), 100)
                        x_hat_reshaped = x_hat.reshape(-1, 1)
                        y_hat = model.predict(x_hat_reshaped)
                        
                        regression_df = pd.DataFrame({
                            "mins": x_hat,
                            f"{self.var}_fit": y_hat,
                            "period": period,
                            "instance": instance
                        })
                        
                        # Use the same dash pattern for all periods
                        instance_regression = alt.Chart(regression_df).mark_line(
                            strokeDash=[2, 2],  # Consistent dash pattern for all periods
                            size=2
                        ).encode(
                            x=x_axis,
                            y=alt.Y(f"{self.var}_fit:Q", scale=y_scale),
                            color=alt.Color("instance:N", scale=alt.Scale(scheme=color_scheme), legend=None),
                            tooltip=alt.value(None),
                        )
                        
                        regression_layers.append(instance_regression)
                        
                    except Exception as e:
                        print(f"Linear regression failed: {e}")

        return regression_layers

    
    def create_trajectory_events(self, instance_df:pd.DataFrame) -> alt.Chart:
        events = alt.Chart(instance_df[instance_df["event_name"].notna()]).mark_point(
            filled=True,
            opacity=0.8,
            size=100,
        ).encode(
            x=alt.X("mins:Q"),
            y=alt.Y(f"{self.var}:Q"),
            color=alt.Color(
                "event_legend:N",  
                scale=alt.Scale(domain=["Event","Calendar Event"], range=["#FFD700","#32CD32"]),  
                legend=alt.Legend(title="Events")
            ),
            tooltip=[
                alt.Tooltip("event_name", title="Event"),
                alt.Tooltip("isoDate:T", title="Time", format=r"%c")
            ]
        ).transform_calculate(
            event_legend="'Event'" 
        )
        calendar = alt.Chart(instance_df[instance_df["calendar_name"].notna()]).mark_point(
            filled=True,
            opacity=0.8,
            size=100,
            color="#32CD32",
        ).encode(
            x=alt.X("mins:Q"),
            y=alt.Y(f"{self.var}:Q"),
            color=alt.Color(
                "event_legend:N",  
                scale=alt.Scale(domain=["Event","Calendar Event"], range=["#FFD700","#32CD32"]),    
                legend=None
            ),
            tooltip=[
                alt.Tooltip("calendar_name", title="Calendar Event"),
                alt.Tooltip("isoDate:T", title="Time", format=r"%c")
            ]
        ).transform_calculate(
            event_legend="'Calendar Event'" 
        )

        return self.combine_layers([events, calendar], resolve="shared")
    
    def plot_aggregate(self):

        # Don't plot anything if no data are found
        if self.aggregate_df.empty:
            return
        
        bars_layers = []
        for effect, color_scheme in zip(['Positive','Negative'], ['blues','reds']):
            effect_differences_df = self.differences_df[self.differences_df["effect"] == effect]
                
            bars = alt.Chart(effect_differences_df).mark_bar().encode(
                x = alt.X("instance:N", axis=alt.Axis(title="Instance", labels=False)),
                y = alt.Y("mean:Q", axis=alt.Axis(title=f"% Change in {self.var_label}", format='%')),
                color = alt.Color("instance:N", scale=alt.Scale(scheme=color_scheme), legend=alt.Legend(title=f"Instance ({effect} Effect)")), #
            ).properties(
                title=alt.Title(f"% Change in {self.var_label} Before and After Intervention", anchor="middle")
            )
            bars_layers.append(bars)

        bars_all = alt.layer(*bars_layers).resolve_scale(color='independent')

        # Render the chart in Streamlit
        st.altair_chart(bars_all, use_container_width=True)

def get_plot(var:str, df:pd.DataFrame):
    """
    Returns an Altair chart for the specified variable and dataframe. 
    """

    df = update_time_segments(df)
    if var == "Stress Level":
        chart = plot_stress_level(df)
    elif var == "Heart Rate":
        chart = plot_heart_rate(df)
    elif var == "Respiration Rate":
        chart = plot_respiration(df)
    elif var == "Beat-to-beat Interval":
        chart = plot_bbi(df)
    elif var == "Steps Taken":
        chart = plot_steps(df)
    return chart

# Define the color ranges based on stress levels
def get_color(stress_level):
    try:
        stress_level = float(stress_level)  # Convert to float to handle numeric strings
        if 0 <= stress_level <= 25:
            return 'dodgerblue'  # Resting state
        elif 26 <= stress_level <= 50:
            return 'gold'  # Low stress
        elif 51 <= stress_level <= 75:
            return 'darkorange'  # Medium stress
        elif 76 <= stress_level <= 100:
            return 'red'  # High stress
        return 'gray'  # Out-of-range values
    except (ValueError, TypeError):
        return 'gray'  # Handle None, non-numeric values, or conversion errors

def get_stress_level_indicator(stress_level):
    """
    Returns a descriptive indicator for a given stress level value.
    
    Args:
        stress_level (float): Stress level value between 0 and 100
        
    Returns:
        str: Description of the stress state
    """
    if 0 <= stress_level <= 25:
        return 'Resting'
    elif 26 <= stress_level <= 50:
        return 'Low Stress'
    elif 51 <= stress_level <= 75:
        return 'Medium Stress'
    elif 76 <= stress_level <= 100:
        return 'High Stress'
    return 'Unknown'  # Default for out-of-range values

def update_time_segments(df:pd.DataFrame, time_gap_threshold:str="5min"):
    """
    Helper function to split non-continuous data points according to the time_gap_threshold.
    """

    # Identify unique segments to separate lines in the plot
    copy_df = df.copy()
    if copy_df["timestamp_cleaned"].isna().sum() > 0:
        copy_df['timestamp_cleaned'] = pd.to_datetime(copy_df['unix_timestamp_cleaned'], unit='ms')
    if "isoDate" not in copy_df.columns:
        copy_df["isoDate"] = pd.to_datetime(copy_df['timestamp_cleaned'])
    copy_df = copy_df.sort_values(by='isoDate')
    copy_df["time_diff"] = copy_df["isoDate"].diff()
    threshold = pd.Timedelta(time_gap_threshold)
    copy_df['segment'] = (copy_df['time_diff'] > threshold).cumsum()
    copy_df = copy_df.drop(columns=["time_diff"])
    return copy_df

def plot_stress_level(stress: pd.DataFrame):
    '''
    Returns an Altair plot of stress levels over time with interactive zoom based on a given selection.
    '''
    # Add a column for color based on stress level
    stress['color'] = stress['stressLevel'].apply(get_color)
    # Add a column for stress indicator
    stress['stress_indicator'] = stress['stressLevel'].apply(get_stress_level_indicator)

    # Get the minimum and maximum dates for the title
    min_date = stress['isoDate'].min()
    max_date = stress['isoDate'].max()

    # Format the dates to include in the title
    formatted_min_date = pd.to_datetime(min_date).strftime('%Y-%m-%d %H:%M')
    formatted_max_date = pd.to_datetime(max_date).strftime('%Y-%m-%d %H:%M')

    # Create the Altair chart
    chart = alt.Chart(stress).mark_rule(opacity=0.7).encode(
        x=alt.X('isoDate:T', title='Timestamp', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=45)),
        y=alt.Y('stressLevel:Q', title='Stress level value'),
        color=alt.Color('stress_indicator:N', 
                       scale=alt.Scale(
                           domain=['Resting', 'Low Stress', 'Medium Stress', 'High Stress', 'Unknown'],
                           range=['dodgerblue', 'gold', 'darkorange', 'red', 'gray']
                       ),
                       legend=alt.Legend(title="Stress State")),
        tooltip=[
            alt.Tooltip('isoDate:T', title='Time', format=r"%c"),
            alt.Tooltip('stressLevel:Q', title='Stress Level'),
            alt.Tooltip('stress_indicator:N', title='Stress State')
        ]
    ).properties(
        width=800,
        height=400,
        title=f'Stress level values over time from {formatted_min_date} to {formatted_max_date}'
    )

    return chart

def get_hr_zone(age, current_hr):
    max_hr = 220 - age  # Max HR formula
    if (0.5 * max_hr) <= current_hr <= (0.59 * max_hr):
        return 1  # Zone 1: For warm-up and recovery
    elif (0.6 * max_hr) <= current_hr <= (0.69 * max_hr):
        return 2  # Zone 2: For aerobic and base fitness
    elif (0.7 * max_hr) <= current_hr <= (0.79 * max_hr):
        return 3  # Zone 3: For aerobic endurance
    elif (0.8 * max_hr) <= current_hr <= (0.89 * max_hr):
        return 4  # Zone 4: For anaerobic capacity
    elif (0.9 * max_hr) <= current_hr <= (max_hr + 15):  # Zone 5: For short burst speed training
        return 5
    else:
        return 0  # Default for out-of-range values

def plot_heart_rate(df: pd.DataFrame):
    '''
    Returns an Altair plot of heart rate over time with interactive zoom based on a given selection.
    '''
    
    # Define heart rate zone colors
    hr_zone_colors = {
        0: 'gray',         # Zone 0: Out of range or resting
        1: 'gray',         # Zone 1: Warm-up
        2: 'dodgerblue',   # Zone 2: Aerobic and base fitness
        3: 'green',        # Zone 3: Aerobic endurance
        4: 'orange',       # Zone 4: Anaerobic capacity
        5: 'red'           # Zone 5: Speed training
    }

    # Calculate and add heart rate zones as a new column
    df['hr_zone'] = df['beatsPerMinute'].apply(lambda hr: get_hr_zone(23, hr))

    # Add a color column based on heart rate zone
    df['color'] = df['hr_zone'].map(hr_zone_colors)

    min_date = df['isoDate'].min()
    max_date = df['isoDate'].max()

    # Format the dates to include in the title
    formatted_min_date = pd.to_datetime(min_date).strftime('%Y-%m-%d %H:%M')
    formatted_max_date = pd.to_datetime(max_date).strftime('%Y-%m-%d %H:%M')

    # Create the Altair chart
    chart = alt.Chart(df).mark_line(opacity=0.7).encode(
        x=alt.X('isoDate:T', title='Timestamp', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=45)),
        y=alt.Y('beatsPerMinute:Q', title='Heart rate (bpm)'),
        detail="segment:N",
        tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"beatsPerMinute:Q", title="Heart rate (bpm)", format=r".2f"),
            ],
        # color=('color:N', 
        #                 legend=alt.Legend(
        #                     title="Heart Rate Zones",
        #                     orient="right",
        #                     titleFontSize=12,
        #                     labelFontSize=10,
        #                     values=['gray', 'dodgerblue', 'green', 'orange', 'red'],
        #                     symbolType='line',
        #                     direction='vertical'
        #                 )
        # )  # Include color legend for heart rate zones
    ).properties(
        width=800,
        height=400,
        title=f'Heart rate (bpm) over time from {formatted_min_date} to {formatted_max_date}'
    )
    return chart  # Return the main chart with the embedded legend

def plot_respiration(respiration: pd.DataFrame):
    '''
    Returns an Altair plot of respiration rate over time with interactive zoom based on a given selection.
    '''
    
    # Get the minimum and maximum dates for the title
    min_date = respiration['isoDate'].min()
    max_date = respiration['isoDate'].max()

    # Format the dates to include in the title
    formatted_min_date = pd.to_datetime(min_date).strftime('%Y-%m-%d %H:%M')
    formatted_max_date = pd.to_datetime(max_date).strftime('%Y-%m-%d %H:%M')

    # Create the Altair chart for respiration rate
    chart = alt.Chart(respiration).mark_line(opacity=0.7).encode(
        x=alt.X('isoDate:T', title='Timestamp', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=45)),
        y=alt.Y('breathsPerMinute:Q', title='Respiration rate (breaths per minute)'),
        color=alt.value('green'),  # Use a single color for respiration rate plot
        detail="segment:N",
        tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"breathsPerMinute:Q", title="Respiration rate", format=r".2f"),
            ],
    ).properties(
        width=800,
        height=400,
        title=f'Respiration rate over time from {formatted_min_date} to {formatted_max_date}'
    )
    return chart

def plot_bbi(bbi: pd.DataFrame) -> alt.Chart:
    # Get the minimum and maximum dates for the title
    min_date = bbi['isoDate'].min()
    max_date = bbi['isoDate'].max()

    # Format the dates to include in the title
    formatted_min_date = pd.to_datetime(min_date).strftime('%Y-%m-%d %H:%M')
    formatted_max_date = pd.to_datetime(max_date).strftime('%Y-%m-%d %H:%M')

    chart = alt.Chart(bbi).mark_line(opacity=0.7).encode(
        x=alt.X('isoDate:T', title='Timestamp', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=45)),
        y=alt.Y('bbi:Q', title='Beat-to-beat interval (ms)', 
                axis=alt.Axis(titlePadding=15)),  # Add padding to prevent cutoff
        color=alt.value('red'),
        detail="segment:N",
        tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"bbi:Q", title="Beat-to-beat interval", format=r".2f"),
            ],
    ).properties(
        width=800,
        height=400,
        title=f'Beat-to-beat interval over time from {formatted_min_date} to {formatted_max_date}'
        # Remove padding property from here
    )

    return chart

def plot_steps(steps: pd.DataFrame) -> alt.Chart:

    # Get the minimum and maximum dates for the title
    min_date = steps['isoDate'].min()
    max_date = steps['isoDate'].max()

    # Format the dates to include in the title
    formatted_min_date = pd.to_datetime(min_date).strftime('%Y-%m-%d %H:%M')
    formatted_max_date = pd.to_datetime(max_date).strftime('%Y-%m-%d %H:%M')

    chart = alt.Chart(steps).mark_rule(opacity=0.7).encode(
        x=alt.X('isoDate:T', title='Timestamp', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=45)),
        y=alt.Y('totalSteps:Q', title='Cumulative Steps Per Day'),
        color=alt.value('lightblue'),
        detail="segment:N",
        tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"totalSteps:Q", title="Cumulative Steps Per Day"),
            ],
    ).properties(
        width=800,
        height=400,
        title=f'Number of steps taken from {formatted_min_date} to {formatted_max_date}'
    )

    return chart