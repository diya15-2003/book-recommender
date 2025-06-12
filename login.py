import streamlit as st
import mysql.connector

# ----------------- DB Connection -----------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Default for WAMP
        database="book_recommender",
        charset='utf8'
    )

# ----------------- User Authentication -----------------
def verify_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def add_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        return True
    except mysql.connector.errors.IntegrityError:
        return False
    finally:
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="login-title">📚 Book Recommender</h1>', unsafe_allow_html=True)

# ----------------- UI Layout -----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="avatar">👤</div>', unsafe_allow_html=True)

    if st.session_state.auth_mode == "login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.switch_page("pages/dashboard.py")
            else:
                st.error("Invalid username or password")

        st.markdown("Don't have an account?")
        if st.button("Sign Up"):
            st.session_state.auth_mode = "signup"

    elif st.session_state.auth_mode == "signup":
        st.subheader("Create a New Account")
        new_user = st.text_input("Choose a Username")
        new_pass = st.text_input("Choose a Password", type="password")

        if st.button("Sign Up"):
            if add_user(new_user, new_pass):
                st.success("Account created! Please log in.")
                st.session_state.auth_mode = "login"
            else:
                st.warning("Username already exists.")

        st.markdown("Already have an account?")
        if st.button("Back to Login"):
            st.session_state.auth_mode = "login"
