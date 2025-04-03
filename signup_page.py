import streamlit as st
import streamlit_analytics
from sql_utils import get_rds_connection
import pymysql

streamlit_analytics.start_tracking()

def check_table_structure():
    """Check the structure of the users table and return column names"""
    conn = get_rds_connection()
    cursor = conn.cursor()
    
    # Check if the users table exists
    cursor.execute("SHOW TABLES LIKE 'users'")
    table_exists = cursor.fetchone()
    
    columns = []
    if table_exists:
        # Get the column names
        cursor.execute("DESCRIBE users")
        columns = [column[0] for column in cursor.fetchall()]
    
    conn.close()
    return columns, table_exists is not None

def create_database():
    # Connect to the RDS database
    conn = get_rds_connection()
    cursor = conn.cursor()

    # SQL query to create the users table if it doesn't exist
    # Use the original column names to maintain compatibility
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) UNIQUE,
            password VARCHAR(255)
        )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def create_custom_database(var_name="events"):
    # Connect to the RDS database
    conn = get_rds_connection()
    cursor = conn.cursor()

    # Create the custom table with a foreign key reference to users.name
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {var_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            start_time BIGINT,
            end_time BIGINT,
            {var_name} TEXT,
            FOREIGN KEY(name) REFERENCES users(name) ON DELETE NO ACTION ON UPDATE NO ACTION
        )
    ''')

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

def insert_user(username, password):
    # Connect to the RDS database
    conn = get_rds_connection()
    cursor = conn.cursor()

    try:
        # Insert the user into the users table using 'name' column
        cursor.execute('''
            INSERT INTO users (name, password)
            VALUES (%s, %s)
        ''', (username, password))
        success = True
    except pymysql.err.OperationalError as e:
        # If 'password' column doesn't exist, try to add it
        if "Unknown column 'password'" in str(e):
            try:
                cursor.execute("ALTER TABLE users ADD password VARCHAR(255)")
                cursor.execute('''
                    INSERT INTO users (name, password)
                    VALUES (%s, %s)
                ''', (username, password))
                success = True
            except Exception as e2:
                st.error(f"Error adding password column: {e2}")
                # Try inserting without password as a fallback
                cursor.execute('''
                    INSERT INTO users (name)
                    VALUES (%s)
                ''', (username,))
                st.warning("Password could not be saved due to database limitations.")
                success = True
        else:
            st.error(f"Database error: {e}")
            success = False
    except Exception as e:
        st.error(f"Error: {e}")
        success = False

    # Commit the transaction and close the connection
    if success:
        conn.commit()
    conn.close()
    return success

def is_username_exists(username):
    # Connect to the RDS database
    conn = get_rds_connection()
    cursor = conn.cursor()

    # Check if a user with the given username exists using 'name' column
    cursor.execute('''SELECT * FROM users WHERE name = %s''', (username,))
    result = cursor.fetchone()

    # Close the connection
    conn.close()
    
    return result is not None

def signup_page():
    create_database()
    create_custom_database("events")
    create_custom_database("interventions")
    
    st.title('Signup')
    
    # Simple form with just username and password
    username = st.text_input('Username', key='username')
    password = st.text_input('Password', type='password', key='password')
    confirm_password = st.text_input('Confirm Password', type='password', key='confirm_password')

    if st.button('Signup'):
        if not username or not password:
            st.warning("Username and password are required.")
        elif password != confirm_password:
            st.warning("Passwords do not match.")
        elif not is_username_exists(username):
            if insert_user(username, password):
                st.success("Signup successful. Please login.")
            else:
                st.error("Failed to create account. Please try again.")
        else:
            st.warning("Username already exists. Please choose a different one.")

if __name__ == '__main__':
    signup_page()