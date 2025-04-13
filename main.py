import streamlit as st
import streamlit_analytics
from signup_page import signup_page
from login_page import login_page
from visualization_page import visualization_page
from data_page import data_upload
from upload_calendar import calendar_upload_page
from download_data import download_user_data
from sql_utils import get_admin_name
from home import home_page
from physiological_analysis import run_stepper_extraction

password = st.secrets["password"]
with streamlit_analytics.track(unsafe_password=password):

    def main():

        st.sidebar.title("Main Menu")
        page = st.sidebar.radio(
            "Go to",
            [
                "Home",
                "Login",
                "Enter Annotations",
                "Show Annotations",
                "Compare Interventions",
                "Calendar Upload",
                "Download Data - Admin",
                "Feature Extractions - Admin",
                "Upload Data - Admin",
                "Signup - Admin",
            ],
        )

        st.session_state.logged_in = st.session_state.get("logged_in", False)

        if page == "Home":
            home_page()
        elif page == "Signup - Admin":
            signup_page()
        elif page == "Login":
            login_successful = login_page()
            if login_successful:
                st.session_state.page = "Upload Data"
        elif page == "Upload Data - Admin":
            if not st.session_state.logged_in:
                st.warning("You must log in to access this page.")
                return
            data_upload()
        elif page == "Calendar Upload":
            if not st.session_state.logged_in:
                st.warning("You must log in to access this page.")
                return
            calendar_upload_page()
        elif page == "Enter Annotations":
            if not st.session_state.logged_in:
                st.warning("You must log in to access this page.")
                return
            visualization_page(annotation=False)
        elif page == "Show Annotations":
            if not st.session_state.logged_in:
                st.warning("You must log in to access this page.")
                return
            visualization_page(annotation=True)
        elif page == "Compare Interventions":
            if not st.session_state.logged_in:
                st.warning("You must log in to access this page.")
                return
            visualization_page(diff_plot=True)
        elif page == "Download Data - Admin":
            if not (
                st.session_state.logged_in and get_admin_name() == st.session_state.user
            ):
                st.warning("You must log in as the admin to access this page.")
                return
            download_user_data()
        elif page == "Feature Extractions - Admin":
            run_stepper_extraction()

    if __name__ == "__main__":
        main()
