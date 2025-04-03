import streamlit as st
import streamlit_analytics
from data_utils import update_database
from sql_utils import *
from sql_utils import get_admin_name
import time

streamlit_analytics.start_tracking()

# Authenticate user by checking against the RDS database
def authenticate_user(name, password):
    conn = get_rds_connection()
    c = conn.cursor()
    
    try:
        # First check if the password column exists
        c.execute("DESCRIBE users")
        columns = [column[0] for column in c.fetchall()]
        
        if 'password' in columns:
            # If password column exists, verify both username and password
            c.execute("SELECT * FROM users WHERE name = %s AND password = %s", (name, password))
            result = c.fetchone()
            
            # If no match with password, check if user exists (wrong password)
            if not result:
                c.execute("SELECT * FROM users WHERE name = %s", (name,))
                user_exists = c.fetchone() is not None
                if user_exists:
                    st.error("Incorrect password. Please try again.")
        else:
            # If password column doesn't exist, only verify username
            c.execute("SELECT * FROM users WHERE name = %s", (name,))
            result = c.fetchone()
            if result:
                st.warning("Password verification is not available. Logging in with username only.")
    except Exception as e:
        st.error(f"Database error: {e}")
        result = None
    finally:
        conn.close()
        
    return result

# Streamlit login page function
def login_page():
    st.title('Login')
    
    # Create a container for the success message
    success_container = st.empty()
    
    name = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if not name:
            st.error("Username is required.")
            return False
            
        # Check if it's the admin account
        if name.strip() == get_admin_name():
            # Show success message
            with success_container.container():
                st.success(f"Welcome, {name.strip()}! Login successful.")
            
            # Set session state
            st.session_state.logged_in = True
            st.session_state.user = name.strip()
            
            # Add a small delay to show the success message
            time.sleep(1.5)
            
            return True
            
        # Otherwise, authenticate with database
        user = authenticate_user(name, password)
        if user:
            # Show success message
            with success_container.container():
                st.success(f"Welcome, {user[1]}! Login successful.")
            
            # Set session state
            st.session_state.logged_in = True
            st.session_state.user = user[1]  # Save the logged-in user's name
            
            # Add a small delay to show the success message
            time.sleep(1.5)
            
            return True  # Indicate successful login
        else:
            if not st.session_state.get('error_shown', False):  # Avoid duplicate error messages
                st.error("Invalid username or password. Please try again.")
                st.session_state.error_shown = True
            return False  # Indicate unsuccessful login

    # Reset error flag when not submitting
    if 'error_shown' in st.session_state:
        del st.session_state.error_shown
        
    return False  # Default: indicate unsuccessful login

# Main function to run the app
if __name__ == '__main__':
    login_page()