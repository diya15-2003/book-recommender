import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import quote
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


# Check login status before showing dashboard
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    # Redirect to login page if not logged in
    st.switch_page("login.py")
    st.stop()

# Page config
st.set_page_config(
    page_title=" BookMind - AI Book Recommender",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .book-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .emotion-chip {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
    }
    
    .joy-chip { background: linear-gradient(45deg, #FFD700, #FFA500); }
    .anger-chip { background: linear-gradient(45deg, #FF4444, #CC0000); }
    .fear-chip { background: linear-gradient(45deg, #800080, #4B0082); }
    .sadness-chip { background: linear-gradient(45deg, #4169E1, #0000CD); }
    .surprise-chip { background: linear-gradient(45deg, #FF69B4, #FF1493); }
    .neutral-chip { background: linear-gradient(45deg, #708090, #2F4F4F); }
    
    .stats-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar-menu-button {
        width: 100%;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border: none;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sidebar-menu-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .sidebar-menu-button.active {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Profile grid smaller styling */
    .profile-book-card {
        max-width: 200px;
        margin: 0.5rem auto;
    }
    
    .profile-book-card img {
        max-height: 150px;
        object-fit: cover;
    }
    
</style>
""", unsafe_allow_html=True)

# User favorites persistence functions
def get_favorites_file_path(username):
    """Get the file path for user's favorites"""
    favorites_dir = "user_favorites"
    if not os.path.exists(favorites_dir):
        os.makedirs(favorites_dir)
    return os.path.join(favorites_dir, f"{username}_favorites.json")

def load_user_favorites(username):
    """Load user's favorite books from file"""
    file_path = get_favorites_file_path(username)
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading favorites: {str(e)}")
    return []

def save_user_favorites(username, favorites):
    """Save user's favorite books to file"""
    file_path = get_favorites_file_path(username)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving favorites: {str(e)}")

# Initialize session state
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None
if 'show_details' not in st.session_state:
    st.session_state.show_details = False
if 'show_profile' not in st.session_state:
    st.session_state.show_profile = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🔍 Search Books"
if 'search_results' not in st.session_state:
    st.session_state.search_results = pd.DataFrame()

# Initialize user favorites from file
username = st.session_state.get('username', 'Guest')
if 'user_choices' not in st.session_state:
    st.session_state.user_choices = load_user_favorites(username)

@st.cache_data
def load_data():
    """Load and preprocess book data"""
    try:
        books = pd.read_csv(r"D:\book recommender\books_with_emotions.csv")
        books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
        books["large_thumbnail"] = np.where(
            books["large_thumbnail"].isna(),
            "cover-not-found.jpg",
            books["large_thumbnail"],
        )
        return books
    except FileNotFoundError:
        st.error("📁 books_with_emotions.csv file not found! Please ensure it's in the same directory.")
        return pd.DataFrame()

@st.cache_resource
def initialize_embeddings():
    """Initialize the embedding model and vector database"""
    try:
        load_dotenv()
        file_path = r"D:\book recommender\tagged_description.txt"
        raw_documents = TextLoader(file_path, encoding='utf-8').load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db_books = Chroma.from_documents(documents, embedding_model)
        return db_books
    except Exception as e:
        st.error(f"🔧 Error initializing embeddings: {str(e)}")
        return None

def get_emotion_color(emotion, value):
    """Get color for emotion visualization"""
    colors = {
        'joy': '#FFD700', 'anger': '#FF4444', 'fear': '#800080',
        'sadness': '#4169E1', 'surprise': '#FF69B4', 'neutral': '#708090'
    }
    return colors.get(emotion, '#708090')

def create_emotion_chart(book_row):
    """Create emotion radar chart for a book"""
    emotions = ['joy', 'anger', 'fear', 'sadness', 'suprise', 'neutral']
    values = [book_row[emotion] for emotion in emotions]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=['Joy', 'Anger', 'Fear', 'Sadness', 'Surprise', 'Neutral'],
        fill='toself',
        name='Emotional Profile',
        line_color='rgb(102, 126, 234)',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=400,
        title="📊 Emotional Profile"
    )
    return fig

def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    """Get book recommendations based on semantic search"""
    if db_books is None:
        return pd.DataFrame()
    
    try:
        recs = db_books.similarity_search(query, k=initial_top_k)
        books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
        book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

        if category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
        else:
            book_recs = book_recs.head(final_top_k)

        # Sort by emotional tone
        if tone == "Happy":
            book_recs = book_recs.sort_values(by="joy", ascending=False)
        elif tone == "Surprising":
            book_recs = book_recs.sort_values(by="suprise", ascending=False)
        elif tone == "Angry":
            book_recs = book_recs.sort_values(by="anger", ascending=False)
        elif tone == "Suspenseful":
            book_recs = book_recs.sort_values(by="fear", ascending=False)
        elif tone == "Sad":
            book_recs = book_recs.sort_values(by="sadness", ascending=False)

        return book_recs
    except Exception as e:
        st.error(f"🔍 Search error: {str(e)}")
        return pd.DataFrame()

def format_authors(authors_str):
    """Format author names properly"""
    if pd.isna(authors_str):
        return "Unknown Author"
    
    authors_split = str(authors_str).split(";")
    if len(authors_split) == 1:
        return authors_split[0]
    elif len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    else:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"

def get_buy_links(title, authors):
    """Generate purchase links for the book"""
    search_query = f"{title} {authors}".replace(" ", "+")
    return {
        "Amazon": f"https://www.amazon.com/s?k={search_query}&i=stripbooks",
        "Goodreads": f"https://www.goodreads.com/search?q={search_query}",
        "Google Books": f"https://www.google.com/search?tbm=bks&q={search_query}"
    }

def display_book_details(book_row):
    """Display detailed book information with improved responsive UI and relevant links"""
    
    with st.container():
        col1, col2 = st.columns([1, 2], gap="medium")

        with col1:
            try:
                st.image(book_row["large_thumbnail"], use_container_width=True, caption=book_row["title"])
            except:
                st.image("https://via.placeholder.com/300x400?text=No+Cover", use_container_width=True)

            fig = create_emotion_chart(book_row)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"## 📖 {book_row['title']}")
            st.markdown(f"**✍️ Author(s):** {format_authors(book_row['authors'])}")
            st.markdown(f"**📅 Published:** {int(book_row['published_year']) if pd.notna(book_row['published_year']) else 'Unknown'}")
            st.markdown(f"**📚 Category:** {book_row['simple_categories']}")
            st.markdown(f"**⭐ Rating:** {book_row['average_rating']:.2f}/5.0 ({int(book_row['ratings_count'])} ratings)")
            st.markdown(f"**📄 Pages:** {int(book_row['num_pages']) if pd.notna(book_row['num_pages']) else 'Unknown'}")

            # Description
            st.markdown("### 📝 Description")
            description = book_row.get('description', '')
            if description and str(description).strip() and str(description) != 'nan':
                st.markdown(
                    f"<div style='background-color: #000000; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; font-style: italic; color:white;'>{description}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("📝 No description available for this book.")

            # Emotional Tone Chips
            st.markdown("### 🎭 Emotional Tone")
            emotions = {
                'Joy': ('joy', '#28a745'), 'Anger': ('anger', '#dc3545'),
                'Fear': ('fear', '#6f42c1'), 'Sadness': ('sadness', '#17a2b8'),
                'Surprise': ('suprise', '#fd7e14'), 'Neutral': ('neutral', '#6c757d')
            }
            chips = []
            for name, (key, color) in emotions.items():
                value = book_row.get(key, 0)
                if value > 0.3:
                    chips.append(
                        f"<span style='display: inline-block; background-color: {color}; color: white; padding: 4px 12px; margin: 2px; border-radius: 20px; font-size: 0.85em;'>{name}: {value:.2f}</span>"
                    )
            if chips:
                st.markdown("".join(chips), unsafe_allow_html=True)
            else:
                st.markdown("*Balanced emotional tone*")

    st.markdown("---")

    # Book links section
    st.markdown("### 📚 Find This Book")

    tab1, tab2 = st.tabs(["🆓 Free Sources", "🛒 Purchase"])

    with tab1:
        st.markdown("#### Legal Free Sources")
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("📖 Project Gutenberg", get_gutenberg_link(book_row['title']),
                           help="Public domain books", use_container_width=True)
            st.link_button("🌐 Internet Archive", get_archive_link(book_row['title']),
                           help="Free digital library", use_container_width=True)
        with col2:
            st.link_button("📚 Open Library", get_openlibrary_link(book_row['title'], format_authors(book_row['authors'])),
                           help="Borrow books online", use_container_width=True)
            st.link_button("🏛️ WorldCat", get_worldcat_link(book_row['title'], format_authors(book_row['authors'])),
                           help="Find in local libraries", use_container_width=True)

    with tab2:
        st.markdown("#### Where to Buy")
        buy_links = get_buy_links(book_row['title'], format_authors(book_row['authors']))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.link_button("🛍️ Amazon", buy_links.get('Amazon', "#"), use_container_width=True)
        with col2:
            st.link_button("📚 Goodreads", buy_links.get('Goodreads', "#"), use_container_width=True)
        with col3:
            st.link_button("🔍 Google Books", buy_links.get('Google Books', "#"), use_container_width=True)


def get_gutenberg_link(title):
    """Generate Project Gutenberg search link"""
    import urllib.parse
    encoded_title = urllib.parse.quote(title)
    return f"https://www.gutenberg.org/ebooks/search/?query={encoded_title}"


def get_openlibrary_link(title, authors):
    """Generate Open Library search link"""
    import urllib.parse
    search_query = f"{title} {authors}".strip()
    encoded_query = urllib.parse.quote(search_query)
    return f"https://openlibrary.org/search?q={encoded_query}"


def get_archive_link(title):
    """Generate Internet Archive search link"""
    import urllib.parse
    encoded_title = urllib.parse.quote(title)
    return f"https://archive.org/search.php?query={encoded_title}&and[]=mediatype%3A%22texts%22"


def get_worldcat_link(title, authors):
    """Generate WorldCat search link to find book in local libraries"""
    import urllib.parse
    search_query = f"{title} {authors}".strip()
    encoded_query = urllib.parse.quote(search_query)
    return f"https://www.worldcat.org/search?q={encoded_query}"


def get_overdrive_link(title):
    """Generate OverDrive search link for digital library books"""
    import urllib.parse
    encoded_title = urllib.parse.quote(title)
    return f"https://www.overdrive.com/search?q={encoded_title}"


def add_book_details_css():
    """Add custom CSS for improved book details styling"""
    st.markdown("""
    <style>
    /* Improve button styling */
    .stButton > button {
        width: 100%;
        margin: 4px 0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 8px 8px 0 0;
    }
    
    /* Link button improvements */
    .stLinkButton > a {
        width: 100%;
        text-decoration: none;
        border-radius: 8px;
    }
    
    /* Emotion chips container */
    .emotion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 10px 0;
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
    }
    
    /* Description box styling */
    .description-box {
        background: linear-gradient(135deg, #000000 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        font-style: italic;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)


def is_book_in_choices(book_isbn):
    """Check if a book is in user's choices"""
    return any(choice.get('isbn13') == book_isbn for choice in st.session_state.user_choices)

def add_to_choices(book):
    """Add a book to user's choices"""
    if not is_book_in_choices(book['isbn13']):
        st.session_state.user_choices.append(book.to_dict())
        save_user_favorites(username, st.session_state.user_choices)

def remove_from_choices(book_isbn):
    """Remove a book from user's choices"""
    st.session_state.user_choices = [
        choice for choice in st.session_state.user_choices 
        if choice.get('isbn13') != book_isbn
    ]
    save_user_favorites(username, st.session_state.user_choices)

def display_book_grid_with_heart(recommendations):
    """Display books in a responsive grid with heart icon for favorites"""
    if recommendations.empty:
        st.warning("🤷‍♀️ No recommendations found. Try adjusting your search terms!")
        return

    cols_per_row = 4
    rows = len(recommendations) // cols_per_row + (1 if len(recommendations) % cols_per_row > 0 else 0)

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            book_idx = row * cols_per_row + col_idx
            if book_idx < len(recommendations):
                book = recommendations.iloc[book_idx]
                with cols[col_idx]:
                    with st.container(border=True):
                        # Display image
                        try:
                            st.image(book["large_thumbnail"], use_container_width=True)
                        except:
                            st.image("https://via.placeholder.com/200x300?text=No+Cover", use_container_width=True)

                        # Title & Author
                        st.markdown(f"**{book['title'][:30]}{'...' if len(book['title']) > 30 else ''}**")
                        author_text = format_authors(book['authors'])
                        st.markdown(f"*{author_text[:25]}{'...' if len(author_text) > 25 else ''}*")

                        # Rating, Pages & Heart
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"⭐ {book['average_rating']:.1f} | 📄 {int(book['num_pages']) if pd.notna(book['num_pages']) else '?'}")

                        with col2:
                            # Heart Icon with faster response
                            is_favorite = is_book_in_choices(book['isbn13'])
                            heart_icon = "❤️" if is_favorite else "🤍"
                            
                            if st.button(heart_icon, key=f"heart_{book['isbn13']}_{book_idx}", help="Add to favorites"):
                                if is_favorite:
                                    remove_from_choices(book['isbn13'])
                                    st.toast("Removed from favorites!", icon="💔")
                                else:
                                    add_to_choices(book)
                                    st.toast("Added to favorites!", icon="❤️")
                                st.rerun()

                        # View Details Button
                        if st.button("📖 View Details", key=f"details_{book['isbn13']}_{book_idx}", use_container_width=True):
                            st.session_state.selected_book = book
                            st.session_state.show_details = True
                            st.rerun()

        # Add separator between rows
        if row < rows - 1:
            st.markdown('<hr style="margin: 1.5rem 0; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)


# Load data
books = load_data()
db_books = initialize_embeddings()

if books.empty:
    st.stop()

# Header
st.markdown("""
<div class="main-header" style="position: relative;">
    <h1 style="margin-bottom:0;">📚 BookMind</h1>
    <h2> AI-Powered Book Recommender</h2>
    <p style="margin-top:0;">Discover your next favorite book using advanced semantic search and emotion analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.markdown("<h2 style='font-weight:bold; text-align:center; margin-bottom:0.5rem;'> DASHBOARD</h2>", unsafe_allow_html=True)
    st.markdown("---")

    # Direct menu buttons
    menu_options = [" Search Books", " Discussions", " Library Stats", " About"]
    
    for option in menu_options:
        button_class = "active" if st.session_state.current_page == option else ""
        if st.button(option, key=f"menu_{option}", use_container_width=True):
            if st.session_state.current_page != option:  # Only change if different
                st.session_state.current_page = option
                st.session_state.show_profile = False
                st.session_state.show_details = False  # Hide details when switching pages
                st.rerun()

    st.markdown("---")

    # User Profile Section
    if st.button(f"👤 {username} Profile", key="profile_btn", use_container_width=True):
        st.session_state.show_profile = True
        st.rerun()

    if st.button("🚪 Logout", use_container_width=True):
        # Save favorites before logout
        save_user_favorites(username, st.session_state.user_choices)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Logged out successfully!")
        st.switch_page("login.py")

# Handle book details view - Display in main dashboard area
if st.session_state.get("show_details", False) and st.session_state.selected_book is not None:
    if st.button("← Back to Search Results", key="back_to_results"):
        st.session_state.show_details = False
        st.session_state.selected_book = None
        st.rerun()
    
    display_book_details(st.session_state.selected_book)

# Main content - only show if not displaying book details
# Enhanced User Profile Section (Replace the existing profile section in your main code)

# Enhanced User Profile Section (Replace the existing profile section in your main code)

elif st.session_state.get("show_profile", False):
    st.header(f"👤 {username}'s Profile")
    
    if st.button("← Back to Dashboard", key="back_to_dashboard"):
        st.session_state.show_profile = False
        st.rerun()
    
    # User Statistics
    if st.session_state.user_choices:
        fav_books = st.session_state.user_choices
        
        # Profile statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Favorite Books", len(fav_books))
        with col2:
            avg_rating = sum(book.get('average_rating', 0) for book in fav_books) / len(fav_books)
            st.metric("⭐ Avg Rating", f"{avg_rating:.2f}")
        with col3:
            total_pages = sum(book.get('num_pages', 0) for book in fav_books if pd.notna(book.get('num_pages', 0)))
            st.metric("📄 Total Pages", f"{int(total_pages):,}")
        with col4:
            categories = set(book.get('simple_categories', 'Unknown') for book in fav_books)
            st.metric("🏷️ Categories", len(categories))
        
        st.markdown("---")
        
        # Display mode toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### ❤️ Your Favorite Books")
        with col2:
            view_mode = st.selectbox("View Mode:", ["Grid View", "Detailed View"], key="profile_view_mode")
        
        if view_mode == "Grid View":
            # Enhanced grid view with more details and actions
            cols_per_row = 4  # Reduced to 4 for better spacing
            rows = len(fav_books) // cols_per_row + (1 if len(fav_books) % cols_per_row > 0 else 0)
            
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    book_idx = row * cols_per_row + col_idx
                    if book_idx < len(fav_books):
                        book = fav_books[book_idx]
                        with cols[col_idx]:
                            with st.container(border=True):
                                # Book cover image
                                try:
                                    st.image(book["large_thumbnail"], use_container_width=True, caption="")
                                except:
                                    st.image("https://via.placeholder.com/150x200?text=No+Cover", use_container_width=True)
                                
                                # Book title and author
                                title_short = book['title'][:25] + "..." if len(book['title']) > 25 else book['title']
                                st.markdown(f"**{title_short}**", help=book['title'])
                                
                                author_short = format_authors(book['authors'])[:20] + "..." if len(format_authors(book['authors'])) > 20 else format_authors(book['authors'])
                                st.markdown(f"*{author_short}*", help=format_authors(book['authors']))
                                
                                # Rating and category
                                st.markdown(f"⭐ {book['average_rating']:.1f} | 📚 {book.get('simple_categories', 'Unknown')[:12]}")
                                
                                # Pages and year
                                pages = int(book['num_pages']) if pd.notna(book.get('num_pages')) else 'Unknown'
                                year = int(book['published_year']) if pd.notna(book.get('published_year')) else 'Unknown'
                                st.markdown(f"📄 {pages} pages | 📅 {year}")
                                
                                # Action buttons
                                if st.button("💔 Remove", key=f"profile_remove_{book['isbn13']}_{book_idx}", 
                                           use_container_width=True, help="Remove from favorites"):
                                    remove_from_choices(book['isbn13'])
                                    st.toast("Removed from favorites!", icon="💔")
                                    st.rerun()
        
        else:  # Detailed View
            # Enhanced detailed view with comprehensive information
            for idx, book in enumerate(fav_books):
                with st.expander(f"📖 {book['title']}", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Book cover
                        try:
                            st.image(book["large_thumbnail"], width=200, caption="")
                        except:
                            st.image("https://via.placeholder.com/200x300?text=No+Cover", width=200)
                        
                        # Quick actions with enhanced functionality
                        col_actions = st.columns(3)
                        
                        with col_actions[0]:
                            if st.button("📖 View Details", key=f"detailed_view_{book['isbn13']}", use_container_width=True):
                                # Convert dict back to pandas Series for display_book_details
                                book_series = pd.Series(book)
                                st.session_state.selected_book = book_series
                                st.session_state.show_details = True
                                st.session_state.show_profile = False
                                st.session_state.details_from_profile = True  # Add this flag
                                st.rerun()
                        
                        with col_actions[1]:
                            if st.button("💔 Remove", key=f"detailed_remove_{book['isbn13']}", use_container_width=True):
                                remove_from_choices(book['isbn13'])
                                st.toast("Removed from favorites!", icon="💔")
                                st.rerun()
                        
                        with col_actions[2]:
                            # Quick purchase link
                            buy_links = get_buy_links(book['title'], format_authors(book['authors']))
                            st.markdown(f"[🛒 Buy]({buy_links['Amazon']})", unsafe_allow_html=True)
                    
                    with col2:
                        # Basic information
                        st.markdown(f"**📚 Title:** {book['title']}")
                        st.markdown(f"**✍️ Author(s):** {format_authors(book['authors'])}")
                        st.markdown(f"**📅 Published:** {int(book['published_year']) if pd.notna(book.get('published_year')) else 'Unknown'}")
                        st.markdown(f"**🏷️ Category:** {book.get('simple_categories', 'Unknown')}")
                        st.markdown(f"**⭐ Rating:** {book['average_rating']:.2f}/5.0 ({int(book.get('ratings_count', 0))} ratings)")
                        st.markdown(f"**📄 Pages:** {int(book['num_pages']) if pd.notna(book.get('num_pages')) else 'Unknown'}")
                        st.markdown(f"**🔢 ISBN:** {book.get('isbn13', 'Not available')}")
                        
                        # Description (truncated)
                        description = book.get('description', 'No description available.')
                        if len(description) > 300:
                            st.markdown(f"**📝 Description:** {description[:300]}...")
                        else:
                            st.markdown(f"**📝 Description:** {description}")
                        
                        # Emotional profile
                        st.markdown("**🎭 Emotional Profile:**")
                        emotions = {
                            'Joy': ('joy', '#28a745'), 
                                    'Anger': ('anger', '#dc3545'),
                                    'Fear': ('fear', '#6f42c1'), 
                                    'Sadness': ('sadness', '#17a2b8'),
                                    'Surprise': ('suprise', '#fd7e14'), 
                                    'Neutral': ('neutral', '#6c757d')
                        }
                        
                        emotion_cols = st.columns(3)
                        for i, (emotion_name, (emotion_key, emoji)) in enumerate(emotions.items()):
                            with emotion_cols[i % 3]:
                                value = book.get(emotion_key, 0)
                                if value > 0.3:  # Only show significant emotions
                                    st.markdown(f"{emoji} {emotion_name}: {value:.2f}")
                        
                        # Purchase links
                        st.markdown("**🛒 Purchase Links:**")
                        buy_links = get_buy_links(book['title'], format_authors(book['authors']))
                        
                        link_cols = st.columns(3)
                        with link_cols[0]:
                            st.markdown(f"[🛍️ Amazon]({buy_links['Amazon']})")
                        with link_cols[1]:
                            st.markdown(f"[📚 Goodreads]({buy_links['Goodreads']})")
                        with link_cols[2]:
                            st.markdown(f"[🔍 Google Books]({buy_links['Google Books']})")
                
                # Add a separator between books
                if idx < len(fav_books) - 1:
                    st.markdown("---")
        
        # Category breakdown chart for user's favorites
        if len(fav_books) > 1:
            st.markdown("---")
            st.markdown("### 📊 Your Reading Preferences")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                categories = [book.get('simple_categories', 'Unknown') for book in fav_books]
                category_counts = pd.Series(categories).value_counts()
                
                fig_cat = px.pie(
                    values=category_counts.values, 
                    names=category_counts.index,
                    title="📚 Favorite Categories",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                # Rating distribution
                ratings = [book.get('average_rating', 0) for book in fav_books]
                
                fig_rating = px.histogram(
                    x=ratings, 
                    nbins=10,
                    title="⭐ Rating Distribution of Your Favorites",
                    color_discrete_sequence=['#667eea']
                )
                fig_rating.update_layout(
                    xaxis_title="Rating",
                    yaxis_title="Number of Books"
                )
                st.plotly_chart(fig_rating, use_container_width=True)
            
            # Emotional profile of user's favorites
            emotion_cols = ['joy', 'anger', 'fear', 'sadness', 'suprise', 'neutral']
            emotion_data = []
            
            for emotion in emotion_cols:
                avg_emotion = sum(book.get(emotion, 0) for book in fav_books) / len(fav_books)
                emotion_data.append(avg_emotion)
            
            fig_emotions = px.bar(
                x=['Joy', 'Anger', 'Fear', 'Sadness', 'Surprise', 'Neutral'],
                y=emotion_data,
                title="🎭 Your Emotional Reading Profile",
                color=emotion_data,
                color_continuous_scale="viridis"
            )
            fig_emotions.update_layout(
                xaxis_title="Emotions",
                yaxis_title="Average Score"
            )
            st.plotly_chart(fig_emotions, use_container_width=True)
            
            # Reading recommendations based on favorites
            st.markdown("### 💡 Personalized Insights")
            
            # Find most common category
            most_common_category = pd.Series(categories).mode()[0] if categories else "Unknown"
            avg_rating = sum(book.get('average_rating', 0) for book in fav_books) / len(fav_books)
            
            # Find dominant emotion
            dominant_emotion_idx = emotion_data.index(max(emotion_data))
            dominant_emotion = ['Joy', 'Anger', 'Fear', 'Sadness', 'Surprise', 'Neutral'][dominant_emotion_idx]
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📚 **Favorite Genre:** {most_common_category}")
                st.info(f"⭐ **Average Rating Preference:** {avg_rating:.2f}/5.0")
            
            with col2:
                st.info(f"🎭 **Emotional Preference:** {dominant_emotion}")
                st.info(f"📖 **Total Books Loved:** {len(fav_books)}")
    
    else:
        st.info("You haven't added any books to your favorites yet. Click the heart ❤️ on a book to add it here!")
        
        # Show some suggestions for getting started
        st.markdown("### 🌟 Getting Started")
        st.markdown("""
        **How to build your reading profile:**
        1. Go to **🔍 Search Books** in the sidebar
        2. Search for books you're interested in
        3. Click the white heart 🤍 to add books to your favorites
        4. Come back here to see detailed information about your collection
        5. Use the **Detailed View** to see comprehensive information about each book
        
        **Your profile will show:**
        - 📊 Reading statistics and preferences
        - 📚 Detailed information about each favorite book
        - 🎭 Your emotional reading profile
        - 💡 Personalized insights based on your choices
        """)

elif st.session_state.current_page == " Search Books":
    # Search interface
    categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_category = st.selectbox("📚 Category:", categories, key="main_category")
    with col2:
        selected_tone = st.selectbox("🎭 Emotional Tone:", tones, key="main_tone")
    with col3:
        num_results = st.slider("📊 Number of Results:", 8, 32, 16, step=4, key="main_num_results")

    query = st.text_input(
        "🔍 What kind of book are you looking for?",
        placeholder="e.g., A thrilling mystery set in Victorian London, A heartwarming story about friendship...",
        help="Describe the type of book you want - themes, settings, emotions, or plot elements",
        key="main_query"
    )

    if st.button(" Find Books", key="main_find_books", use_container_width=True):
        if query.strip():
            with st.spinner("🔍 Searching for your perfect books..."):
                # Always search for maximum results (32) to allow slider flexibility
                recommendations = retrieve_semantic_recommendations(
                    query, selected_category, selected_tone, final_top_k=32
                )
                st.session_state.full_search_results = recommendations
                st.session_state.has_searched = True
        else:
            st.warning("Please enter a search query to find books!")

    # Display search results if they exist
    if 'has_searched' in st.session_state and st.session_state.has_searched and 'full_search_results' in st.session_state:
        if not st.session_state.full_search_results.empty:
            # Always slice based on current slider value
            current_results = st.session_state.full_search_results.head(num_results)
            
            st.markdown(f"### 📚 Found {len(current_results)} recommendations for you:")
            display_book_grid_with_heart(current_results)
    elif 'search_results' in st.session_state and st.session_state.search_results.empty and query:
        st.warning("No books found matching your criteria. Try different keywords!")

elif st.session_state.current_page == " Discussions":
    st.title("Book Discussions and Fan Theories")
    
    import pandas as pd
    from datetime import datetime
    import os
    import urllib.parse
    
    # Load discussions
    if os.path.exists("discussions.csv"):
        discussions_df = pd.read_csv("discussions.csv")
    else:
        discussions_df = pd.DataFrame(columns=["timestamp", "username", "book_title", "message"])
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .discussion-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .book-title {
        font-size: 18px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 8px;
    }
    .username {
        font-size: 14px;
        color: #7f8c8d;
        font-weight: 500;
    }
    .timestamp {
        font-size: 12px;
        color: #95a5a6;
        font-style: italic;
    }
    .message {
        font-size: 15px;
        color: #34495e;
        margin: 10px 0;
        line-height: 1.5;
    }
    .discussion-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #ffffff;
    }
    .forum-links {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .forum-link-button {
        display: inline-block;
        padding: 8px 16px;
        margin: 5px;
        background-color: #007bff;
        color: white;
        text-decoration: none;
        border-radius: 4px;
        font-size: 14px;
    }
    .forum-link-button:hover {
        background-color: #0056b3;
        text-decoration: none;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Tabs for discussion panel
    tab1, tab2 = st.tabs(["Share Your Thoughts", "View Community Theories"])
    
    # Tab 1: Submit a new discussion
    with tab1:
        with st.form("new_discussion"):
            st.subheader("Start a Discussion or Share a Fan Theory")
            
            col1, col2 = st.columns(2)
            with col1:
                book_title = st.text_input("Book Title", placeholder="Enter the book title...")
            with col2:
                username = st.text_input("Your Name", 
                                       value=st.session_state.username if "username" in st.session_state else "",
                                       placeholder="Your display name...")
            
            message = st.text_area("Share your thoughts, theories, or questions about the book...", 
                                 height=120,
                                 placeholder="What did you think about the plot? Any interesting theories? Favorite characters?")
            
            submit = st.form_submit_button("Post Discussion", use_container_width=True)
            
            if submit:
                if not book_title.strip() or not username.strip() or not message.strip():
                    st.error("Please fill in all fields.")
                else:
                    new_entry = pd.DataFrame([{
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "username": username.strip(),
                        "book_title": book_title.strip(),
                        "message": message.strip()
                    }])
                    
                    discussions_df = pd.concat([discussions_df, new_entry], ignore_index=True)
                    discussions_df.to_csv("discussions.csv", index=False)
                    st.success("Your discussion has been posted successfully!")
                    st.rerun()
    
    # Tab 2: View discussions
    with tab2:
        st.subheader("Community Discussions")
        
        # Sort options only
        if not discussions_df.empty:
            sort_options = ["Newest First", "Oldest First"]
            sort_order = st.selectbox("Sort by:", sort_options)
        
        # Display discussions
        if discussions_df.empty:
            st.info("No discussions yet. Be the first to start one!")
        else:
            # Show all discussions
            filtered_discussions = discussions_df
            
            if filtered_discussions.empty:
                st.warning("No discussions found.")
            else:
                # Sort discussions
                ascending = False if sort_order == "Newest First" else True
                sorted_discussions = filtered_discussions.sort_values("timestamp", ascending=ascending)
                
                st.markdown(f"**{len(sorted_discussions)} discussion(s)**")
                
                # Create scrollable container using Streamlit components
                with st.container(height=500):
                    for _, row in sorted_discussions.iterrows():
                        with st.container():
                            # Discussion card styling using columns and markdown
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin: 0px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <h4 style="color: #2c3e50; margin-bottom: 8px;">{row['book_title']}</h4>
                                <p style="color: #34495e; margin: 0px 0; line-height: 2;">{row['message']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Username and timestamp in columns
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.caption(f"by {row['username']}")
                            with col2:
                                st.caption(row['timestamp'])
                            
                            st.divider()
                
                # BookClubs.com link at the end
                st.markdown("**For more further discussions, look at this page or connect with more people: [BookClubs.com](https://bookclubs.com/discussions)")
        

elif st.session_state.current_page == " Library Stats":
    st.header(" Library Statistics")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Total Books", f"{len(books):,}")
    with col2:
        st.metric("📖 Avg Rating", f"{books['average_rating'].mean():.2f}")
    with col3:
        st.metric("📄 Avg Pages", f"{int(books['num_pages'].mean()):,}")
    with col4:
        st.metric("🏷️ Categories", books['simple_categories'].nunique())
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        category_counts = books['simple_categories'].value_counts().head(10)
        fig_cat = px.bar(
            x=category_counts.values, y=category_counts.index,
            orientation='h', title="📚 Books by Category",
            color=category_counts.values, color_continuous_scale="viridis"
        )
        fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        fig_rating = px.histogram(
            books, x='average_rating', nbins=20,
            title="⭐ Rating Distribution",
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig_rating, use_container_width=True)
    
    # Emotion analysis
    emotion_cols = ['joy', 'anger', 'fear', 'sadness', 'suprise', 'neutral']
    emotion_means = books[emotion_cols].mean()
    
    fig_emotions = px.bar(
        x=emotion_means.index, y=emotion_means.values,
        title="🎭 Average Emotional Content Across All Books",
        color=emotion_means.values, color_continuous_scale="plasma"
    )
    st.plotly_chart(fig_emotions, use_container_width=True)

elif st.session_state.current_page == " About":
    st.header("ℹ️ About BookMind")
    
    st.markdown("""
    ### 🤖 About BookMind

    BookMind is an AI-powered book recommendation system that uses advanced natural language processing 
    to understand your reading preferences and find books that match your interests.

    **🔧 Features:**
    - **Semantic Search**: Describe what you want to read in natural language
    - **Emotion Analysis**: Books are analyzed for emotional content (joy, sadness, fear, etc.)
    - **Category Filtering**: Browse by genre and category
    - **Favorites System**: Save books to your personal collection with the heart ❤️ button
    - **Detailed Information**: View descriptions, ratings, and purchase links
    - **Interactive UI**: Modern, responsive design for the best user experience
    - **Persistent Favorites**: Your favorite books are saved and will persist across sessions

    **🧠 How it works:**
    1. Book descriptions are processed using sentence transformers
    2. Your query is embedded in the same semantic space
    3. Similar books are found using vector similarity
    4. Results are filtered and sorted by your preferences

    **💡 Tips for better results:**
    - Be descriptive about themes, settings, or emotions you want
    - Try different emotional tones to discover new books
    - Use the category filter to narrow down results
    - Click the heart ❤️ to save books to your favorites
    - Visit your profile to see all your saved books
    - Your favorites are automatically saved and will be there when you log back in

    **❤️ Using Favorites:**
    - Click the white heart 🤍 on any book to add it to your favorites
    - The heart will turn red ❤️ immediately to show it's saved
    - Click the red heart ❤️ again to remove from favorites
    - View all your favorites in your profile section
    - Your favorites are saved to your account and persist across login sessions

    Built with ❤️ using Streamlit, LangChain, and Hugging Face Transformers.
    """)
