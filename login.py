import streamlit as st
import mysql.connector
import re
import hashlib
import time
from mysql.connector import Error

# ----------------- Configuration -----------------
MIN_USERNAME_LENGTH = 3
MAX_USERNAME_LENGTH = 20
MIN_PASSWORD_LENGTH = 6
MAX_PASSWORD_LENGTH = 50
MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_TIME = 300  # 5 minutes in seconds

# ----------------- Validation Functions -----------------
def validate_username(username):
    """Validate username format and length"""
    errors = []
    
    if not username:
        errors.append("Username is required")
        return errors
    
    if len(username) < MIN_USERNAME_LENGTH:
        errors.append(f"Username must be at least {MIN_USERNAME_LENGTH} characters long")
    
    if len(username) > MAX_USERNAME_LENGTH:
        errors.append(f"Username must be no more than {MAX_USERNAME_LENGTH} characters long")
    
    # Only allow alphanumeric characters and underscores
    if not re.match("^[a-zA-Z0-9_]+$", username):
        errors.append("Username can only contain letters, numbers, and underscores")
    
    # Username cannot start with a number
    if username[0].isdigit():
        errors.append("Username cannot start with a number")
    
    return errors

def validate_password(password):
    """Validate password strength and format"""
    errors = []
    
    if not password:
        errors.append("Password is required")
        return errors
    
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
    
    if len(password) > MAX_PASSWORD_LENGTH:
        errors.append(f"Password must be no more than {MAX_PASSWORD_LENGTH} characters long")
    
    # Check for at least one uppercase letter
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")
    
    # Check for at least one lowercase letter
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")
    
    # Check for at least one digit
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one number")
    
    # Check for at least one special character
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)")
    
    return errors

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_login_attempts():
    """Check if user has exceeded login attempts"""
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
        st.session_state.last_failed_attempt = 0
    
    current_time = time.time()
    
    # Reset attempts if lockout time has passed
    if current_time - st.session_state.last_failed_attempt > LOCKOUT_TIME:
        st.session_state.login_attempts = 0
    
    return st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS

def record_failed_login():
    """Record a failed login attempt"""
    st.session_state.login_attempts += 1
    st.session_state.last_failed_attempt = time.time()

def reset_login_attempts():
    """Reset login attempts after successful login"""
    st.session_state.login_attempts = 0
    st.session_state.last_failed_attempt = 0

# ----------------- DB Connection -----------------
def get_connection():
    """Get database connection with error handling"""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Default for WAMP
            database="book_recommender",
            charset='utf8',
            autocommit=False,  # Explicit transaction control
            connection_timeout=10  # Connection timeout
        )
        return conn
    except Error as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

# ----------------- User Authentication -----------------
def verify_user(username, password):
    """Verify user credentials with proper error handling"""
    if not username or not password:
        return False
    
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor(prepared=True)  # Use prepared statements
        hashed_password = hash_password(password)
        
        query = "SELECT id FROM users WHERE username = ? AND password = ?"
        cursor.execute(query, (username, hashed_password))
        result = cursor.fetchone()
        
        return result is not None
    
    except Error as e:
        st.error(f"Login verification failed: {str(e)}")
        return False
    
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def check_username_exists(username):
    """Check if username already exists"""
    conn = get_connection()
    if not conn:
        return True  # Assume exists if can't connect (safer)
    
    try:
        cursor = conn.cursor(prepared=True)
        query = "SELECT id FROM users WHERE username = ?"
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        
        return result is not None
    
    except Error as e:
        st.error(f"Username check failed: {str(e)}")
        return True  # Assume exists if error occurs
    
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def add_user(username, password):
    """Add new user with proper validation and error handling"""
    conn = get_connection()
    if not conn:
        return False, "Database connection failed"
    
    try:
        cursor = conn.cursor(prepared=True)
        hashed_password = hash_password(password)
        
        query = "INSERT INTO users (username, password) VALUES (?, ?)"
        cursor.execute(query, (username, hashed_password))
        conn.commit()
        
        return True, "Account created successfully"
    
    except mysql.connector.errors.IntegrityError:
        return False, "Username already exists"
    
    except Error as e:
        conn.rollback()
        return False, f"Failed to create account: {str(e)}"
    
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# ----------------- Session Initialization -----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# ----------------- Already Logged In? -----------------
if st.session_state.logged_in:
    st.switch_page("pages/dashboard.py")  # Skip login screen

# ----------------- UI Style -----------------
st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(-45deg, #f8ffae, #43cea2, #185a9d, #f5576c);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .login-title {
        color: #fff;
        text-shadow: 1px 2px 12px #185a9d66;
        font-weight: 800;
        text-align: center;
    }
    .avatar {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        margin: 0 auto 1.2rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        color: #fff;
        background: linear-gradient(135deg, #f093fb 0%, #43cea2 100%);
        box-shadow: 0 2px 12px #185a9d33;
        border: 3px solid #43cea2;
    }
    .validation-error {
        color: #ff4444;
        font-size: 0.85em;
        margin-top: 0.2rem;
    }
    .validation-success {
        color: #44ff44;
        font-size: 0.85em;
        margin-top: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="login-title">📚 Book Recommender</h1>', unsafe_allow_html=True)

# ----------------- UI Layout -----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="avatar">👤</div>', unsafe_allow_html=True)

    # Check if user is locked out
    if check_login_attempts():
        remaining_time = LOCKOUT_TIME - (time.time() - st.session_state.last_failed_attempt)
        if remaining_time > 0:
            st.error(f"Too many failed login attempts. Please try again in {int(remaining_time/60)} minutes and {int(remaining_time%60)} seconds.")
            st.stop()

    if st.session_state.auth_mode == "login":
        st.subheader("Login")
        
        # Use form to enable Enter key submission
        with st.form("login_form"):
            username = st.text_input("Username", max_chars=MAX_USERNAME_LENGTH)
            password = st.text_input("Password", type="password", max_chars=MAX_PASSWORD_LENGTH)
            
            # Form submit button - this will be triggered by Enter key
            login_submitted = st.form_submit_button("Login", use_container_width=True)

        if login_submitted:
            # Basic validation for empty fields
            if not username.strip():
                st.error("Please enter your username")
            elif not password:
                st.error("Please enter your password")
            else:
                # Attempt login
                if verify_user(username.strip(), password):
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    reset_login_attempts()
                    st.success("Login successful!")
                    time.sleep(1)  # Brief pause to show success message
                    st.switch_page("pages/dashboard.py")
                else:
                    record_failed_login()
                    remaining_attempts = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                    if remaining_attempts > 0:
                        st.error(f"Invalid username or password. {remaining_attempts} attempts remaining.")
                    else:
                        st.error("Account temporarily locked due to too many failed attempts.")

        st.markdown("---")
        st.markdown("Don't have an account?")
        if st.button("Sign Up"):
            st.session_state.auth_mode = "signup"
            st.rerun()

    elif st.session_state.auth_mode == "signup":
        st.subheader("Create a New Account")
        
        # Use form for signup to enable Enter key submission
        with st.form("signup_form", clear_on_submit=False):
            new_user = st.text_input("Choose a Username", max_chars=MAX_USERNAME_LENGTH, 
                                    help="Press Enter after filling all fields to create account")
            new_pass = st.text_input("Choose a Password", type="password", max_chars=MAX_PASSWORD_LENGTH,
                                   help="Must contain uppercase, lowercase, number, and special character")
            confirm_pass = st.text_input("Confirm Password", type="password", max_chars=MAX_PASSWORD_LENGTH,
                                       help="Re-enter your password")
            
            # Real-time validation inside form (will update on form interaction)
            if new_user:
                username_errors = validate_username(new_user)
                if username_errors:
                    for error in username_errors:
                        st.markdown(f'<div class="validation-error">❌ {error}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-success">✅ Username format is valid</div>', unsafe_allow_html=True)

            if new_pass:
                password_errors = validate_password(new_pass)
                if password_errors:
                    for error in password_errors:
                        st.markdown(f'<div class="validation-error">❌ {error}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-success">✅ Password meets requirements</div>', unsafe_allow_html=True)

            if confirm_pass and new_pass:
                if new_pass != confirm_pass:
                    st.markdown('<div class="validation-error">❌ Passwords do not match</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-success">✅ Passwords match</div>', unsafe_allow_html=True)
            
            # Form submit button - responds to Enter key from any field
            signup_submitted = st.form_submit_button("Sign Up", use_container_width=True)

        if signup_submitted:
            # Validate all inputs
            username_errors = validate_username(new_user)
            password_errors = validate_password(new_pass)
            
            if username_errors:
                st.error("Username validation failed:")
                for error in username_errors:
                    st.error(f"• {error}")
            elif password_errors:
                st.error("Password validation failed:")
                for error in password_errors:
                    st.error(f"• {error}")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match")
            elif check_username_exists(new_user):
                st.error("Username already exists. Please choose a different username.")
            else:
                # All validations passed, create account
                success, message = add_user(new_user, new_pass)
                if success:
                    st.success(message + " Please log in.")
                    time.sleep(2)  # Brief pause to show success message
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(message)

        st.markdown("---")
        st.markdown("Already have an account?")
        if st.button("Back to Login"):
            st.session_state.auth_mode = "login"
            st.rerun()

    # Display password requirements
    if st.session_state.auth_mode == "signup":
        with st.expander("Password Requirements"):
            st.markdown("""
            **Password must contain:**
            - At least 6 characters
            - At least one uppercase letter (A-Z)
            - At least one lowercase letter (a-z)
            - At least one number (0-9)
            - At least one special character (!@#$%^&*(),.?\":{}|<>)
            
            **Username requirements:**
            - 3-20 characters long
            - Only letters, numbers, and underscores
            - Cannot start with a number
            """)