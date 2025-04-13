import streamlit as st
import pandas as pd
import altair as alt
from dataclasses import dataclass, field
from vega_datasets import data
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
import io
import base64
import uuid

def df_to_altair(df:pd.DataFrame):

    # df = df.copy()  # work on a copy to avoid modifying the original DataFrame

    df_records = df.to_dict(orient="records")

    cleaned_records = []

    for record in df_records:
        new_record = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                new_value = value.isoformat() if pd.notnull(value) else None
            elif isinstance(value, pd.Timedelta):
                new_value = value.total_seconds() / 60
            elif pd.isnull(value):
                new_value = None
            else:
                new_value = value
            new_record[key] = new_value
        cleaned_records.append(new_record)

    # Assign unique key
    unique_key = str(uuid.uuid4())
    
    # Return the data as an Altair inline data object
    return alt.Data(name=unique_key, values=cleaned_records)

def create_moon_sleep_indicator(sleep_score, sleep_timing, size=100, effect_color=None):
    """
    Creates a horizontally filled circle (moon shape) based on a sleep_score from 0..100.
    
    Labeling uses Garmin thresholds for Sleep Quality:
        - >= 90 => "Excellent"
        - >= 80 => "Good"
        - >= 60 => "Fair"
        - < 60  => "Poor"
    
    Color logic:
      - If effect_color is 'blues', fill with #6EB9F7
      - If effect_color is 'reds', fill with #EF476F
      - Otherwise, color is based on the same Garmin thresholds:
          >= 90 => #6EB9F7 (blue)
          >= 80 => #64C2A6 (teal/green)
          >= 60 => #FFD166 (yellow)
          <  60 => #EF476F (red)
    
    If sleep_score is None or invalid, draws a fully gray circle with "No Sleep Data".
    
    Args:
        sleep_score (float): Sleep score in [0..100], or None/NaN
        size (int): (Optional) figure size in pixels (not heavily used here)
        effect_color (str): If provided, overrides color scheme with 'blues' or 'reds'.
    
    Returns:
        str: Base64-encoded PNG suitable for embedding (e.g., in Altair).
    """

    # Create figure
    fig, ax = plt.subplots(figsize=(2.5, 3), dpi=150)
    fig.patch.set_alpha(0)  # transparent background
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    moon_radius = 0.7

    # Helper function: Garmin threshold -> label
    def garmin_label(score):
        if score >= 90:
            return "Excellent Sleep"
        elif score >= 80:
            return "Good Sleep"
        elif score >= 60:
            return "Fair Sleep"
        else:
            return "Poor Sleep"

    # Helper function: Garmin threshold -> color
    def garmin_color(score):
        if score >= 90:
            return '#6EB9F7'  # excellent (blue)
        elif score >= 80:
            return '#64C2A6'  # good (teal/green)
        elif score >= 60:
            return '#FFD166'  # fair (yellow)
        else:
            return '#EF476F'  # poor (red)

    # Check if invalid sleep_score
    if (sleep_score is None or pd.isna(sleep_score)
        or not isinstance(sleep_score, (int, float))):
        # Gray filled circle
        circle = Circle((0, 0), moon_radius, facecolor='lightgray',
                        edgecolor='gray', linewidth=1)
        ax.add_patch(circle)
        ax.text(0, -1.3, "No Sleep Data", ha='center', va='center',
                fontsize=14, color='black', fontweight='bold')
    else:
        # Clamp to [0..100]
        sleep_score = max(0, min(sleep_score, 100))
        fill_level = sleep_score / 100.0  # fraction for partial fill

        # Determine the label from Garmin thresholds
        label = garmin_label(sleep_score)

        # Decide color: if effect_color is forced, override. Otherwise, Garmin color.
        if effect_color == 'blues':
            fill_color = '#6EB9F7'
        elif effect_color == 'reds':
            fill_color = '#EF476F'
        elif effect_color == 'purples':
            fill_color = '#928FBF'
        else:
            fill_color = garmin_color(sleep_score)

        # Outline circle
        outline = Circle((0, 0), moon_radius, facecolor='none',
                         edgecolor='gray', linewidth=1)
        ax.add_patch(outline)

        # Horizontal fill from left to right
        rect_left   = -moon_radius
        rect_bottom = -moon_radius
        rect_height = 2 * moon_radius
        fill_width  = 2 * moon_radius * fill_level

        fill_rect = plt.Rectangle((rect_left, rect_bottom),
                                  fill_width, rect_height,
                                  facecolor=fill_color, edgecolor=None)
        ax.add_patch(fill_rect)

        # Clip rectangle to circle
        circle_path = Path.circle((0, 0), moon_radius)
        fill_rect.set_clip_path(PathPatch(circle_path, transform=ax.transData))

        # Description
        ax.text(0, -1, f"{sleep_timing} Night:", ha="center", va="center",
                fontsize=14, color=fill_color, fontweight="bold")

        # Label
        ax.text(0, -1.3, label, ha="center", va="center",
                fontsize=14, color=fill_color, fontweight="bold")

    # Convert figure to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=150)
    plt.close(fig)
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"
@dataclass
class ComparisonPlotsManager:
    """
    """
    # Attributes of the class
    instances_df: pd.DataFrame
    aggregate_df: pd.DataFrame
    instance_type: str
    var: str
    var_to_label: dict
    mins_before: int
    mins_after: int
    split_comparison_plots: bool = False
    max_duration: float = 0.
    var_label: str = ""
    differences_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    scheme_domains: dict = field(default_factory=dict)

    def __post_init__(self):
        # Only display valid data
        self.instances_df = self.instances_df[self.instances_df[self.var] > 0]
        # Don't plot anything if no data are found
        if self.instances_df.empty:
            st.write(f"There are no valid data for {self.var_to_label[self.var].lower()} for any intervention.")
            return
        # Initialize neat variable name for labelling
        self.var_label = self.var.title().replace("_"," ") if self.var.lower() not in ["bbi","rmssd"] else self.var.upper()
        # Get duration of longest intervention
        self.max_duration = (self.instances_df["duration"].dt.total_seconds() / 60).max()
        # Update segments to prevent interpolation of far points
        self.instances_df = update_time_segments(self.instances_df)
        # Add effect column to dataframes
        self.update_effect()
        # Sort dataframes
        self.update_instance_order()
        # Update instance labels
        self.update_instance_labels()
        # Plot line plots with each instance trajectory
        self.plot_trajectories()
        # Plot bar plot with aggregated data
        self.plot_aggregate()

    @staticmethod
    def combine_layers(layers:list[alt.Chart], resolve:str="independent", prop_kwargs:dict=None) -> alt.Chart:
        """Helper function to combine altair chart layers."""
        plot = alt.layer(
            *layers
        ).resolve_scale(
            color=resolve,
            stroke=resolve,
            theta=resolve
        )

        if prop_kwargs is not None:
            plot = plot.properties(**prop_kwargs) 

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
        elif self.var in ["bbi","rmssd"]:
            # For bbi, increase is positive
            self.differences_df.loc[self.differences_df["mean"] > 0,"effect"] = "Positive"
            self.differences_df.loc[self.differences_df["mean"] <= 0,"effect"] = "Negative"
        else:
            self.differences_df["effect"] = "Neutral"

        # Update instances_df
        for instance in self.differences_df["instance"].unique():
            self.instances_df.loc[self.instances_df["instance"] == instance, "effect"] = self.differences_df[self.differences_df["instance"] == instance]["effect"].iloc[0]
            self.instances_df.loc[self.instances_df["instance"] == instance, "pct_change"] = self.differences_df[self.differences_df["instance"] == instance]["mean"].iloc[0]

    def update_instance_order(self) -> None:
        """
        Method that sorts the instance data according to its date and time.
        """
        # Convert instance name to datetime
        instances_dt = pd.to_datetime(self.instances_df["instance"].unique(), format="%a %d %b %Y, %I:%M%p")
        self.instance_order = [dt.strftime("%a %d %b %Y, %I:%M%p") for dt in sorted(instances_dt)]

    def update_instance_labels(self) -> None:
        """
        Method that adds an "instance_label" field to each dataframe.
        """
        self.instances_df["instance_label"] = self.instances_df["instance"].dt.strftime(r"%a %d %b, %I:%M%p").str.cat(self.instances_df.apply(
            lambda row: f' ({row["pct_change"]:.2%} increase)' if row["pct_change"] > 0 else f' ({-row["pct_change"]:.2%} decrease)', axis=1
        ))
        self.differences_df["instance_label"] = self.differences_df["instance"].dt.strftime(r"%a %d %b, %I:%M%p").str.cat(self.differences_df.apply(
            lambda row: f' ({row["mean"]:.2%} increase)' if row["mean"] > 0 else f' ({-row["mean"]:.2%} decrease)', axis=1
        ))
        
        # Update effects color scheme
        self.scheme_domains = {"Positive":[], "Negative":[], "Neutral":[]}
        for effect in self.scheme_domains.keys():
            self.scheme_domains[effect].extend(list(self.differences_df.loc[self.differences_df["effect"] == effect, "instance_label"]))

    def plot_trajectories(self):
        """
        Method to plot the trajectory of instances as lines.
        Modified for better space usage and effect grouping with improved spacing.
        """
        # Get minimum and maximum of the variable
        var_min, var_max = self.instances_df[self.var].min(),  self.instances_df[self.var].max()

        # Create custom x axis and y scale
        x_axis = self.create_trajectory_x_axis()
        y_span = var_max - var_min
        y_padding = y_span * 0.1
        y_scale = alt.Scale(domain=[var_min-y_padding, var_max+y_padding])

        # Add labels
        labels_layers = self.create_trajectory_annotations()

        # Initialize list of Altair layers
        all_line_layers, all_sleep_layers, all_regression_layers, all_event_layers = [], [], [], []
        # Initialize list of subplots 
        all_subplots = []

        # Adjust properties based on view mode
        if self.split_comparison_plots:
            # Original heights for split view
            default_height = 200
            plot_spacing = 10  # Default spacing
        else:
            # More compact heights for effect-grouped view
            default_height = 250  # Slightly shorter
            plot_spacing = 20  # Positive spacing to create gap between plots
        
        # Initialize inclusion of title
        default_props = dict(
            width=800,
            height=default_height
        )

        effect_color_scale = {k: alt.Scale(scheme=s, domain=v) for s, (k,v) in zip(["blues","reds","purples"], self.scheme_domains.items())}
        
        # Create separate plots
        if self.split_comparison_plots:
            # When in split view, create one plot per instance (original behavior)
            dfs = [self.instances_df[self.instances_df["instance"] == instance] for instance in self.instance_order]
        else:
            # When in combined view, create one plot per effect type (modified behavior)
            dfs = []
            for effect in ["Positive", "Negative", "Neutral"]:  # Ensure consistent order
                effect_df = self.instances_df[self.instances_df["effect"] == effect]
                if not effect_df.empty:
                    dfs.append(effect_df)

        # Iterate through dataframes and plot each one
        for i, instance_df in enumerate(dfs):
            # Skip if dataframe is empty
            if instance_df.empty:
                continue
            # Define the conditional color logic for sleep data
            effect = instance_df["effect"].iloc[0]
            # If effect could not be calculated, skip the plot
            if pd.isna(effect):
                continue
            instance_name = instance_df['instance_label'].iloc[0]
            instances_to_process = [instance_name] if self.split_comparison_plots else instance_df["instance_label"].unique()
            # Aggregate data for sleep reporting
            sleep_steps_df = instance_df.sort_values("mins").groupby("instance_label").aggregate(
                mins_last=("mins", "last"),
                var_first=(self.var, "first"),
                var_last=(self.var, "last"),
                steps_yesterday=("steps_yesterday", "last"),
                today_sleep_score=("today_sleep_score", "last"),
                today_sleep_label=("today_sleep_label", "last"),
                tmr_sleep_score=("tmr_sleep_score", "last"),
                tmr_sleep_label=("tmr_sleep_label", "last"),
                effect_first=("effect", "first"),
            ).reset_index() 
            # Plot the various components of the plot
            instance_line_layers = self.create_trajectory_lines(instance_df, effect, effect_color_scale[effect], x_axis, y_scale)
            instance_sleep_step_layer = self.create_trajectory_sleep_steps(sleep_steps_df, effect_color_scale[effect], y_span)
            instance_regression_layers = self.create_trajectory_regression(instance_df, instances_to_process, x_axis, y_scale, effect_color_scale[effect])
            instance_event_layer = self.create_trajectory_events(instance_df)
            
            # Create props for this specific plot
            subplot_props = default_props.copy()
                        
            # For effect-grouped mode (not split view)
            if not self.split_comparison_plots:
                if i == 0:  # First plot (Positive Effect)
                    subplot_props["title"] = alt.Title(
                        f"{self.var_label} - {effect} Effect",
                        anchor="middle"
                    )
                else:  # Second plot (Negative Effect)
                    # Add offset to the title to create more space
                    subplot_props["title"] = alt.Title(
                        f"{effect} Effect",
                        anchor="middle",
                        offset=15  # Increase title offset to add more space
                    )
            else:
                # Modified title behavior for split view
                if i == 0:
                    subplot_props["title"] = alt.Title(f"{self.var_label} Before, During, and After {self.instance_type.title()}")
                else:
                    subplot_props["title"] = alt.Title("") # Remove instance name from title
            
            # Create the subplot
            subplot = self.combine_layers(
                [*labels_layers, *instance_line_layers, instance_sleep_step_layer, instance_regression_layers, instance_event_layer],
                prop_kwargs=subplot_props
            )
            all_subplots.append(subplot)

        # Create the final vertical concatenation of plots
        if len(all_subplots) > 0:
            # Create a vertical concatenation with POSITIVE spacing
            plot = alt.vconcat(*all_subplots, spacing=plot_spacing).resolve_scale(color="shared", stroke="shared").resolve_legend(color="independent", stroke="independent")
            
            # Render the plots
            st.altair_chart(plot, use_container_width=True)
            
    def create_trajectory_x_axis(self) -> alt.X:
        """
        Creates and returns a custom Altair x-axis for the trajectory plot.
        """
        custom_tick_vals = [i for i in range(0,self.mins_before,10)]
        custom_tick_vals.extend([self.mins_before+i for i in range(0, int(self.max_duration), 10)])
        custom_tick_vals.extend([self.mins_before+int(self.max_duration)+i for i in range(0,self.mins_after+11,10)])
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
                """,
                labelPadding=5,
                titlePadding=5,
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
        data = df_to_altair(labels_df)
        section_labels = alt.Chart(data).mark_text(
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
        data = df_to_altair(pd.DataFrame({
            "x": [self.mins_before, self.mins_before + self.max_duration]
        }))
        vbar_marks = alt.Chart(data).mark_rule(color="red").encode(x="x:Q",tooltip=alt.value(None))
        return [section_labels, vbar_marks]
    
    def create_trajectory_lines(self, df:pd.DataFrame, effect:str, effect_scale:alt.Scale, 
                                x_axis:alt.X, y_scale:alt.Scale) -> tuple[alt.Chart, alt.Chart]:
        """
        Creates and returns Altair chart layers for lines representing trajectories of instances.
        """
        data = df_to_altair(df)

        lines_before_after = alt.Chart(data).mark_line(
            opacity=0.85
        ).encode(
            x=x_axis,
            y=alt.Y(f"{self.var}:Q", axis=alt.Axis(title=self.var_label), scale=y_scale),
            color=alt.Color('instance_label:N', scale=effect_scale, 
                            legend=alt.Legend(title=f'Instance ({effect} Effect)', orient="top", direction="vertical", labelLimit=400, values=df["instance_label"].unique()),
                            sort=self.instance_order),
            detail=alt.Detail(['status:N', 'segment:N', 'instance_label:N']),
            tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"{self.var}:Q", title=self.var_label.title()),
                alt.Tooltip("instance_label:N", title="Instance")
            ]
        ).transform_filter(
            (
                (alt.datum.status == "before") | 
                (alt.datum.status == "after")
            ) & 
            (alt.datum.event_name == None) & 
            (alt.datum.calendar_name == None)
        )
        lines_during = alt.Chart(data).mark_line(
            opacity=0.85
        ).encode(
            x=x_axis,
            y=alt.Y(f"{self.var}:Q", scale=y_scale),
            color=alt.Color('instance_label:N', scale=effect_scale, legend=None),
            detail=alt.Detail('instance_label:N'),
            tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"{self.var}:Q", title=self.var_label.title()),
                alt.Tooltip("instance_label:N", title="Instance"),
            ]
        ).transform_filter(
            alt.datum.status == "during"
        )
        return lines_before_after, lines_during
    
    def create_trajectory_sleep_steps(self, df:pd.DataFrame, effect_scale:alt.Scale, y_span:float, inner_radius=6, outer_radius=12, x_scale:float=0.05, y_scale:float=0.1) -> alt.Chart:
        """
        Creates a separate sleep and steps plot
        """
        shared_x_max, shared_x_min = self.instances_df["mins"].max(), self.instances_df["mins"].min()
        x_offset = (shared_x_max - shared_x_min) * x_scale
        # Create padding variables
        y_offset = y_span * y_scale
        df["x1"] = -x_offset
        df["x2"] = shared_x_max+x_offset
        df["y1"] = df["var_first"]
        df["y2"] = df["var_first"] + y_offset
        df["today_moon_image"] = df.apply(lambda row: create_moon_sleep_indicator(row.get("today_sleep_score"), "Previous", effect_color=effect_scale.to_dict()["scheme"]), axis=1)
        df["tmr_moon_image"] = df.apply(lambda row: create_moon_sleep_indicator(row.get("tmr_sleep_score"), "Next", effect_color=effect_scale.to_dict()["scheme"]), axis=1)
        # Convert to Altair data and plot
        data = df_to_altair(df)
        today_moon_chart = alt.Chart(data).mark_image(
            width=80,
            height=100
        ).encode(
            x = alt.X("x1:Q"),
            y = alt.Y("y1:Q"),
            url="today_moon_image:N",
            tooltip=[
                alt.Tooltip("instance_label:N", title="Instance"),
                alt.Tooltip("today_sleep_label:N", title="Sleep Quality"),
                alt.Tooltip("today_sleep_score:Q", title="Sleep Score")
            ]
        )
        tmr_moon_chart = alt.Chart(data).mark_image(
            width=80,
            height=100
        ).encode(
            x = alt.X("x2:Q"),
            y = alt.Y("y1:Q"),
            url="tmr_moon_image:N",
            tooltip=[
                alt.Tooltip("instance_label:N", title="Instance"),
                alt.Tooltip("tmr_sleep_label:N", title="Sleep Quality"),
                alt.Tooltip("tmr_sleep_score:Q", title="Sleep Score")
            ]
        )
        return self.combine_layers([today_moon_chart, tmr_moon_chart])
    
    def create_trajectory_regression(self, instance_df:pd.DataFrame, 
                                    instances_to_process:list[str], 
                                    x_axis:alt.X, y_scale:alt.Scale, effect_scale:alt.Scale,
                                    regression_method:str="simple") -> alt.Chart:
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
                period_data = instance_df[(instance_df['instance_label'] == instance) & 
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
                            "instance_label": instance
                        })
                        # Create a highlighter effect with dashed lines
                        
                        data = df_to_altair(regression_df)

                        # Wide, very transparent background (no dashes)
                        wide_background = alt.Chart(data).mark_line(
                            strokeWidth=7,  # Very wide for highlighter effect
                            opacity=0.2      # Very transparent
                        ).encode(
                            x=x_axis,
                            y=alt.Y(f"{self.var}_fit:Q", scale=y_scale),
                            color=alt.Color("instance_label:N", scale=effect_scale, legend=None),
                            tooltip=alt.value(None)
                        )
                        
                        # Medium background with wide dashes
                        medium_background = alt.Chart(data).mark_line(
                            strokeWidth=5,    # Medium width
                            opacity=0.3,      # Medium transparency
                            strokeDash=[4, 4] # Wide dashes
                        ).encode(
                            x=x_axis,
                            y=alt.Y(f"{self.var}_fit:Q", scale=y_scale),
                            color=alt.Color("instance_label:N", scale=effect_scale, legend=None),
                            tooltip=alt.value(None)
                        )
                        
                        # Main dashed line on top
                        main_line = alt.Chart(data).mark_line(
                            strokeWidth=2.3,    # Thicker main line
                            opacity=0.8,      # Slightly transparent
                            strokeDash=[4, 4] # Standard dashes
                        ).encode(
                            x=x_axis,
                            y=alt.Y(f"{self.var}_fit:Q", scale=y_scale),
                            color=alt.Color("instance_label:N", scale=effect_scale, legend=None),
                            tooltip=[
                                alt.Tooltip("instance_label:N", title="Instance"),
                                alt.Tooltip("period:N", title="Period"),
                                alt.Tooltip(f"{self.var}_fit:Q", title=self.var_label, format=".1f")
                            ]
                        )
                        
                        # Add all three layers to create a translucent highlighter effect over dashed lines
                        regression_layers.append(wide_background)
                        regression_layers.append(medium_background)
                        regression_layers.append(main_line)
                        
                    except Exception as e:
                        print(f"Linear regression failed: {e}")

        return self.combine_layers(regression_layers, resolve="shared")
    
    def create_trajectory_events(self, instance_df:pd.DataFrame) -> alt.Chart:

        # Initialize empty event lists
        instance_types = []
        instance_colors = []
        event_df_list = []

        # Iterate and filter through all event types
        for instance, instance_color in zip(["Intervention","Event", "Calendar"], ["#F745DA","#FFD700","#32CD32"]):
            instance_mask = instance_df[f"{instance.lower().replace(' ', '_')}_name"].notna()
            # Only add to the lists if there are valid entries for the specific event in the dataframe
            if instance_mask.sum() > 0:
                instance_types.append(instance)
                instance_colors.append(instance_color)
                event_df_list.append(
                    instance_df.loc[instance_mask, ["mins", self.var, f"{instance.lower()}_name", 
                                                    f"{instance.lower()}_start", f"{instance.lower()}_end"
                                                    ]].assign(event_legend=instance)
                )
        
        # Return an empty chart if no valid events were found
        if len(event_df_list) == 0:
            return alt.Chart(pd.DataFrame()).mark_point() 
        
        # Concatenate all events into one dataframe and plot in a single chart layer
        all_events_df = pd.concat(event_df_list)
        data = df_to_altair(all_events_df)
        color_scale = alt.Scale(domain=instance_types, range=instance_colors)
        event_chart = alt.Chart(data).mark_point(
            filled=True, opacity=0.8, size=100
        ).encode(
            x=alt.X("mins:Q"),
            y=alt.Y(f"{self.var}:Q"),
            color=alt.Color("event_legend:N", scale=color_scale, legend=alt.Legend(title="")),
            tooltip=[
                alt.Tooltip(f"{instance.lower()}_name:N", title=instance) for instance in instance_types
            ] + [
                alt.Tooltip(f"{instance.lower()}_start:T", title="Start Time", format=r"%c") for instance in instance_types
            ] + [
                alt.Tooltip(f"{instance.lower()}_end:T", title="End Time", format=r"%c") for instance in instance_types
            ]
        )
        
        return event_chart
    
    def plot_aggregate(self):

        # Don't plot anything if no data are found
        if self.differences_df.empty:
            return
        
        bar_charts = []
        first_plot = True
        for effect, color_scheme in zip(['Positive','Negative'], ['blues','reds']):
            effect_differences_df = self.differences_df[self.differences_df["effect"] == effect]
            if effect_differences_df.empty: continue

            data = df_to_altair(effect_differences_df)

            bars = alt.Chart(data).mark_bar()
            if first_plot:
                bars = bars.properties(title=alt.Title(f"% Change in {self.var_label} Before and After {self.instance_type.title()}", anchor="middle"))
                y_axis_title = ""
                first_plot = False
            else:
                y_axis_title = f"% Change in {self.var_label}"
            
            bars = bars.encode(
                y = alt.Y("instance_label:N", axis=alt.Axis(title="", labels=False), sort=self.instance_order),
                x = alt.X("mean:Q", axis=alt.Axis(title=y_axis_title, format='%')),
                color = alt.Color("instance_label:N", scale=alt.Scale(scheme=color_scheme), legend=alt.Legend(title=f"Instance ({effect} Effect)", orient="top", direction="vertical", labelLimit=400), sort=self.instance_order),
                tooltip=[
                    alt.Tooltip('instance_label:N', title='Instance'),
                    alt.Tooltip('mean:Q', title=f'% Change in {self.var_label}', formatType="number", format=".2%"),
                ]
            )

            bar_charts.append(bars)

        if len(bar_charts) == 0: return

        # Render the chart in Streamlit
        chart = alt.vconcat(*bar_charts).resolve_scale(color="independent", x="shared")
        st.altair_chart(chart, use_container_width=True)

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
    elif var == "Heart Rate Variability":
        chart = plot_rmssd(df)
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
    chart = alt.Chart(stress).mark_rule(opacity=0.8).encode(
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
    chart = alt.Chart(df).mark_line(opacity=0.8).encode(
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
    chart = alt.Chart(respiration).mark_line(opacity=0.8).encode(
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

    chart = alt.Chart(bbi).mark_line(opacity=0.8).encode(
        x=alt.X('isoDate:T', title='Timestamp', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=45)),
        y=alt.Y('bbi:Q', title='Beat-to-beat interval (ms)', 
                axis=alt.Axis(titlePadding=15)),  # Add padding to prevent cutoff
        color=alt.value('tomato'),
        detail="segment:N",
        tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"bbi:Q", title="Beat-to-beat interval", format=r".2f"),
            ],
    ).properties(
        width=800,
        height=400,
        title=f'Beat-to-beat interval over time from {formatted_min_date} to {formatted_max_date}'
    )

    return chart

def plot_rmssd(rmssd: pd.DataFrame) -> alt.Chart:
    # Get the minimum and maximum dates for the title
    min_date = rmssd['isoDate'].min()
    max_date = rmssd['isoDate'].max()

    # Format the dates to include in the title
    formatted_min_date = pd.to_datetime(min_date).strftime('%Y-%m-%d %H:%M')
    formatted_max_date = pd.to_datetime(max_date).strftime('%Y-%m-%d %H:%M')

    # Create altair chart
    chart = alt.Chart(rmssd).mark_line(opacity=0.8).encode(
        x=alt.X('isoDate:T', title='Timestamp', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=45)),
        y=alt.Y('rmssd:Q', title='RMSSD (ms)', 
                axis=alt.Axis(titlePadding=15)),  # Add padding to prevent cutoff
        color=alt.value('teal'),
        detail="segment:N",
        tooltip=[
                alt.Tooltip("isoDate:T", title="Time", format=r"%c"),
                alt.Tooltip(f"rmssd:Q", title="RMSSD (ms)", format=r".2f"),
            ],
    ).properties(
        width=800,
        height=400,
        title=f'Heart Rate Variability over time from {formatted_min_date} to {formatted_max_date}'
    )
    
    return chart 

def plot_steps(steps: pd.DataFrame) -> alt.Chart:

    # Get the minimum and maximum dates for the title
    min_date = steps['isoDate'].min()
    max_date = steps['isoDate'].max()

    # Format the dates to include in the title
    formatted_min_date = pd.to_datetime(min_date).strftime('%Y-%m-%d %H:%M')
    formatted_max_date = pd.to_datetime(max_date).strftime('%Y-%m-%d %H:%M')

    chart = alt.Chart(steps).mark_rule(opacity=0.8).encode(
        x=alt.X('isoDate:T', title='Timestamp', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=45)),
        y=alt.Y('totalSteps:Q', title='Cumulative Steps Per Day'),
        color=alt.value('sienna'),
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