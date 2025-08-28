import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon (first time only)
nltk.download('vader_lexicon')

# Function to fetch tweets
def fetch_tweets(username, limit=20, delay=1):
    tweets_list = []
    query = f"from:{username}"
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            tweets_list.append([tweet.date, tweet.content])
            time.sleep(delay)  # avoid rate blocks
            if i + 1 >= limit:
                break
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
    df = pd.DataFrame(tweets_list, columns=['Datetime', 'Content'])
    return df

# Function for sentiment analysis
def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Content'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['Label'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
    return df

# Streamlit UI
st.title("Twitter Sentiment Analysis (snscrape)")

username = st.text_input("Enter Twitter handle (without @):")
limit = st.number_input("Number of tweets to fetch:", min_value=5, max_value=100, value=20, step=5)

if st.button("Fetch & Analyze"):
    if username:
        with st.spinner("Fetching tweets..."):
            df = fetch_tweets(username, limit=limit)
            if not df.empty:
                df = analyze_sentiment(df)
                st.success(f"Fetched and analyzed {len(df)} tweets from @{username}")
                st.dataframe(df)

                # Sentiment distribution
                st.bar_chart(df['Label'].value_counts())

                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download CSV", data=csv, file_name=f"{username}_tweets_sentiment.csv", mime="text/csv")
            else:
                st.warning("No tweets found or failed to fetch.")
    else:
        st.warning("Please enter a Twitter handle.")
