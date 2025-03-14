import streamlit as st
from signup_page import signup_page
from login_page import login_page
from visualization_page import visualization_page
from data_page import data_upload
from upload_calendar import calendar_upload_page 
from download_data import download_user_data
from sql_utils import get_admin_name
# from log import log_page
# from sql_utils import get_rds_connection


def main():
    st.sidebar.title('Main Menu')
    page = st.sidebar.radio('Go to', [
        'Signup', 
        'Login',    
        'Enter Annotations',         # Changed from 'Visualization'
        'Show Annotations',      # Changed from 'Annotations'
        'Compare Interventions',     # Changed from 'Compare plot'
        'Calendar Upload', 
        'Upload Data - Admin',
        'Download Data - Admin'          # Changed from "Download data"
    ])

    st.session_state.logged_in = st.session_state.get('logged_in', False)

    if page == 'Signup':
        signup_page()
    elif page == 'Login':
        login_successful = login_page()
        if login_successful:
            st.session_state.page = 'Upload Data'  # Changed redirect to Upload Data
    elif page == 'Upload Data - Admin':  # Changed from "Upload data"
        if not st.session_state.logged_in:
            st.warning("You must log in to access this page.")
            return
        data_upload()
    elif page == 'Calendar Upload':
        if not st.session_state.logged_in:
            st.warning("You must log in to access this page.")
            return
        calendar_upload_page()
    elif page == 'Enter Annotations':  # Changed from 'Visualization'
        if not st.session_state.logged_in:
            st.warning("You must log in to access this page.")
            return
        visualization_page(annotation=False)
    elif page == 'Show Annotations':  # Changed from 'Annotations'
        if not st.session_state.logged_in:
            st.warning("You must log in to access this page.")
            return
        visualization_page(annotation=True)
    elif page == 'Compare Interventions':  # Changed from 'Compare plot'
        if not st.session_state.logged_in:
            st.warning("You must log in to access this page.")
            return
        visualization_page(diff_plot=True)
    elif page == 'Download Data - Admin':  # Changed from "Download data"
        if not (st.session_state.logged_in and get_admin_name() == st.session_state.user):
            st.warning("You must log in as the admin to access this page.")
            return
        download_user_data()

if __name__ == '__main__':
    main()