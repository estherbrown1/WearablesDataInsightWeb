import streamlit as st
from signup_page import signup_page
from login_page import login_page
from visualization_page import visualization_page
from data_page import data_upload
from upload_calendar import calendar_upload_page
from download_data import download_user_data
from sql_utils import get_admin_name
from home import home_page
from physiological_analysis import run_stepper_extraction 


from analytics import inject_google_analytics, track_page_view


def main():
   # 1) Inject Google Analytics once
   inject_google_analytics()
  
   st.sidebar.title('Main Menu')
   page = st.sidebar.radio('Go to', [
       'Home',
       'Login',
       'Enter Annotations',
       'Show Annotations',
       'Compare Interventions',
       'Calendar Upload',
       'Download Data - Admin',
       'Feature Extractions - Admin',
       'Upload Data - Admin',
       'Signup - Admin',
   ])


   # 2) Track page view whenever page changes
   track_page_view(page)


   # The rest of the app
   st.session_state.logged_in = st.session_state.get('logged_in', False)


   if page == 'Home':
       home_page()
   elif page == 'Signup - Admin':
       signup_page()
   elif page == 'Login':
       login_successful = login_page()
       if login_successful:
           st.session_state.page = 'Upload Data'
   elif page == 'Upload Data - Admin':
       if not st.session_state.logged_in:
           st.warning("You must log in to access this page.")
           return
       data_upload()
   elif page == 'Calendar Upload':
       if not st.session_state.logged_in:
           st.warning("You must log in to access this page.")
           return
       calendar_upload_page()
   elif page == 'Enter Annotations':
       if not st.session_state.logged_in:
           st.warning("You must log in to access this page.")
           return
       visualization_page(annotation=False)
   elif page == 'Show Annotations':
       if not st.session_state.logged_in:
           st.warning("You must log in to access this page.")
           return
       visualization_page(annotation=True)
   elif page == 'Compare Interventions':
       if not st.session_state.logged_in:
           st.warning("You must log in to access this page.")
           return
       visualization_page(diff_plot=True)
   elif page == 'Download Data - Admin':
       if not (st.session_state.logged_in and get_admin_name() == st.session_state.user):
           st.warning("You must log in as the admin to access this page.")
           return
       download_user_data()
   elif page == 'Feature Extractions - Admin':
       run_stepper_extraction()  


if __name__ == '__main__':
   main()