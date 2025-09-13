import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources (happens automatically)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set up the page
st.set_page_config(
    page_title="MCA Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 MCA Public Comment Sentiment Analysis")
st.markdown("Analyze public sentiment on MCA draft legislation")

# Initialize all session state variables
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'comment_column' not in st.session_state:
    st.session_state.comment_column = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = "Upload Data"
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'page_size' not in st.session_state:
    st.session_state.page_size = 50  # Default page size

# Load sentiment analyzer with caching
@st.cache_resource
def load_sentiment_model():
    st.info("Loading sentiment analysis model...")
    return pipeline("sentiment-analysis")

# Function to display paginated dataframe
def display_paginated_df(df, page_size=50):
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Page selector
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="page_selector")
    
    # Calculate start and end indices
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    # Display page info
    st.write(f"Showing comments {start_idx + 1} to {end_idx} of {total_rows}")
    
    # Display the current page
    current_page_df = df.iloc[start_idx:end_idx]
    st.dataframe(current_page_df, use_container_width=True)
    
    return page, total_pages

# Sidebar for navigation
st.sidebar.title("Navigation")
steps = ["Upload Data", "Clean Data", "Analyze Sentiment", "View Results", "Word Cloud"]
current_step = st.sidebar.radio("Go to", steps, index=steps.index(st.session_state.current_step))

# Update current step based on sidebar selection
if current_step != st.session_state.current_step:
    st.session_state.current_step = current_step
    st.rerun()

# Step 1: Upload Data
if st.session_state.current_step == "Upload Data":
    st.header("📤 Step 1: Upload Your Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file with comments", type="csv")
    
    if uploaded_file is not None:
        try:
            # Try reading with different encodings if needed
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            st.success("✅ File uploaded successfully!")
            st.write(f"📊 Dataset shape: {df.shape}")
            
            # Let user select which column has comments
            comment_column = st.selectbox("Select the column that contains comments", df.columns)
            st.session_state.comment_column = comment_column
            
            st.write("👀 First 50 comments:")
            display_paginated_df(df[[comment_column]].head(50).rename(columns={comment_column: 'Comment'}), page_size=10)
            
            if st.button("Next: Clean Data →"):
                st.session_state.current_step = "Clean Data"
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Try uploading a different file or check the file format.")

# Step 2: Clean Data
elif st.session_state.current_step == "Clean Data" and st.session_state.df is not None:
    st.header("🧹 Step 2: Clean the Data")
    
    df = st.session_state.df
    comment_column = st.session_state.comment_column
    
    # Check if we've already cleaned the data
    if st.session_state.df_clean is None:
        with st.spinner("Cleaning data... This may take a while for large datasets."):
            # Make a copy
            df_clean = df.copy()
            
            # Rename column
            df_clean.rename(columns={comment_column: 'original_comment'}, inplace=True)
            
            # Remove empty comments and duplicates
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=['original_comment'])
            df_clean = df_clean.drop_duplicates(subset=['original_comment'])
            after_clean_count = len(df_clean)
            
            # Cleaning function
            def clean_text(text):
                if not isinstance(text, str):
                    return ""
                try:
                    text = text.lower()
                    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    tokens = word_tokenize(text)
                    stop_words = set(stopwords.words('english'))
                    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
                    return ' '.join(filtered_tokens)
                except:
                    return text
            
            # Apply cleaning with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cleaned_comments = []
            for i, comment in enumerate(df_clean['original_comment']):
                cleaned_comments.append(clean_text(comment))
                
                # Update progress every 100 comments to avoid slowing down
                if i % 100 == 0 or i == len(df_clean) - 1:
                    progress = (i + 1) / len(df_clean)
                    progress_bar.progress(progress)
                    status_text.text(f"Cleaning {i + 1}/{len(df_clean)} comments...")
            
            df_clean['cleaned_comment'] = cleaned_comments
            st.session_state.df_clean = df_clean
            
            progress_bar.empty()
            status_text.empty()
            
            st.info(f"Removed {initial_count - after_clean_count} empty or duplicate comments")
    
    df_clean = st.session_state.df_clean
    st.success(f"✅ Cleaning complete! {len(df_clean)} comments ready for analysis.")
    
    # Show before/after examples with pagination
    st.subheader("Sample of Cleaned Comments")
    sample_size = min(10, len(df_clean))
    sample_df = df_clean[['original_comment', 'cleaned_comment']].head(sample_size)
    sample_df = sample_df.rename(columns={
        'original_comment': 'Original Comment',
        'cleaned_comment': 'Cleaned Comment'
    })
    
    for i, row in sample_df.iterrows():
        with st.expander(f"Comment {i+1}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original:**")
                st.write(row['Original Comment'])
            with col2:
                st.write("**Cleaned:**")
                st.write(row['Cleaned Comment'])
    
    if st.button("Next: Analyze Sentiment →"):
        st.session_state.current_step = "Analyze Sentiment"
        st.rerun()

# Step 3: Analyze Sentiment
elif st.session_state.current_step == "Analyze Sentiment" and st.session_state.df_clean is not None:
    st.header("😊 Step 3: Analyze Sentiment")
    
    df_clean = st.session_state.df_clean
    
    # Load the sentiment analyzer if not already loaded
    if st.session_state.sentiment_analyzer is None:
        with st.spinner("Loading AI model... This may take a few minutes"):
            st.session_state.sentiment_analyzer = load_sentiment_model()
    
    if not st.session_state.analysis_complete:
        if st.button("Start Sentiment Analysis", type="primary"):
            sentiment_analyzer = st.session_state.sentiment_analyzer
            
            # Function to get sentiment
            def get_sentiment(text):
                try:
                    result = sentiment_analyzer(text[:512])[0]
                    label = result['label']
                    score = result['score']
                    if label == 'POSITIVE': 
                        return 'Positive', score
                    if label == 'NEGATIVE': 
                        return 'Negative', score
                    return 'Neutral', score
                except:
                    return 'Neutral', 0.5
            
            # Create placeholders for progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            sentiments = []
            confidences = []
            
            # Analyze all comments with progress bar
            for i, text in enumerate(df_clean['cleaned_comment']):
                label, score = get_sentiment(text)
                sentiments.append(label)
                confidences.append(score)
                
                # Update progress every 10 comments to avoid slowing down the UI
                if i % 10 == 0 or i == len(df_clean) - 1:
                    progress = (i + 1) / len(df_clean)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i + 1}/{len(df_clean)} comments...")
                    
                    # Show intermediate results every 100 comments
                    if i % 100 == 0 and i > 0:
                        temp_df = df_clean.iloc[:i].copy()
                        temp_df['sentiment'] = sentiments[:i]
                        temp_df['confidence'] = confidences[:i]
                        sentiment_counts = temp_df['sentiment'].value_counts()
                        
                        with results_container.container():
                            st.write("**Intermediate Results:**")
                            for sentiment, count in sentiment_counts.items():
                                st.write(f"- {sentiment}: {count} comments")
            
            # Add results to dataframe
            df_clean['sentiment'] = sentiments
            df_clean['confidence'] = confidences
            st.session_state.df_clean = df_clean
            st.session_state.analysis_complete = True
            
            progress_bar.empty()
            status_text.empty()
            results_container.empty()
            
            st.success("✅ Sentiment analysis complete!")
            st.rerun()
    else:
        st.success("✅ Sentiment analysis already complete!")
        
        # Show quick results
        sentiment_counts = df_clean['sentiment'].value_counts()
        st.write("📊 Quick Results:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df_clean)) * 100
            st.write(f"- {sentiment}: {count} comments ({percentage:.1f}%)")
    
    if st.session_state.analysis_complete and st.button("Next: View Detailed Results →"):
        st.session_state.current_step = "View Results"
        st.rerun()

# Step 4: View Results
elif st.session_state.current_step == "View Results" and st.session_state.df_clean is not None:
    st.header("📋 Step 4: View Results")
    
    df_clean = st.session_state.df_clean
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comments", len(df_clean))
    with col2:
        avg_confidence = df_clean['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
    with col3:
        dominant_sentiment = df_clean['sentiment'].mode()[0] if len(df_clean) > 0 else "N/A"
        dominant_count = (df_clean['sentiment'] == dominant_sentiment).sum()
        st.metric("Dominant Sentiment", f"{dominant_sentiment} ({dominant_count})")
    with col4:
        # Let user adjust page size
        st.session_state.page_size = st.selectbox(
            "Comments per page", 
            options=[10, 20, 50, 100], 
            index=2,  # Default to 50
            key="page_size_selector"
        )
    
    # Sentiment distribution chart
    st.subheader("Sentiment Distribution")
    sentiment_counts = df_clean['sentiment'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4CAF50', '#FF5252', '#FFC107']  # Green, Red, Yellow
    wedges, texts, autotexts = ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=colors[:len(sentiment_counts)],
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Sentiment Analysis Results')
    st.pyplot(fig)
    
    # Filter options
    st.subheader("Filter Comments")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_sentiments = st.multiselect(
            "Select sentiments to show:",
            options=df_clean['sentiment'].unique(),
            default=df_clean['sentiment'].unique()
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum confidence:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
    
    # Apply filters
    filtered_df = df_clean[
        (df_clean['sentiment'].isin(selected_sentiments)) & 
        (df_clean['confidence'] >= min_confidence)
    ].copy()
    
    st.write(f"**Showing {len(filtered_df)} of {len(df_clean)} comments**")
    
    if len(filtered_df) > 0:
        # Display paginated results
        page, total_pages = display_paginated_df(
            filtered_df[['original_comment', 'sentiment', 'confidence']].rename(columns={
                'original_comment': 'Comment',
                'sentiment': 'Sentiment',
                'confidence': 'Confidence'
            }),
            page_size=st.session_state.page_size
        )
    else:
        st.warning("No comments match your filters.")
    
    # Download buttons
    st.subheader("Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_clean.to_csv(index=False)
        st.download_button(
            "📥 Download Full Results as CSV",
            data=csv,
            file_name="mca_sentiment_full_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        csv_filtered = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Results as CSV",
            data=csv_filtered,
            file_name="mca_sentiment_filtered_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    if st.button("Next: View Word Cloud →"):
        st.session_state.current_step = "Word Cloud"
        st.rerun()

# Step 5: Word Cloud
elif st.session_state.current_step == "Word Cloud" and st.session_state.df_clean is not None:
    st.header("☁️ Step 5: Word Cloud")
    
    df_clean = st.session_state.df_clean
    
    # Option to filter by sentiment for word cloud
    sentiment_option = st.selectbox(
        "Select sentiment for word cloud:",
        options=["All"] + list(df_clean['sentiment'].unique())
    )
    
    if sentiment_option == "All":
        text_data = ' '.join(df_clean['cleaned_comment'].tolist())
        title = "Most Frequent Words in All Comments"
    else:
        text_data = ' '.join(df_clean[df_clean['sentiment'] == sentiment_option]['cleaned_comment'].tolist())
        title = f"Most Frequent Words in {sentiment_option} Comments"
    
    # Generate word cloud
    if text_data.strip():  # Only generate if there's text data
        with st.spinner("Generating word cloud..."):
            wordcloud = WordCloud(
                width=1000, 
                height=500, 
                background_color='white',
                max_words=100
            ).generate(text_data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16)
            st.pyplot(fig)
        
        st.info("💡 Larger words appear more frequently in the comments")
    else:
        st.warning("No text data available to generate word cloud.")
    
    if st.button("Start Over"):
        # Reset the application
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# If user tries to access steps without completing previous ones
else:
    if st.session_state.current_step != "Upload Data":
        st.warning("Please start by uploading your data in Step 1")
        if st.button("Go to Upload Data"):
            st.session_state.current_step = "Upload Data"
            st.rerun()