import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from transformers import BartForConditionalGeneration, BartTokenizer
from summa import keywords
from bs4 import BeautifulSoup
import requests
import pytz
from datetime import datetime, timedelta
import dateutil.parser
import string
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import numpy as np

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load BART model and tokenizer for text summarization
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def fetch_news_data(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    data_content_titles = []
    timestamps = []
    content_texts = []

    for i in range(1, 2):
        page_url = f"{url}?pgno={i}#Latest"
        r = requests.get(page_url)
        soup = BeautifulSoup(r.text, "html.parser")

        h2_tags = soup.find_all("h2", class_="f18")
        for h2_tag in h2_tags:
            a_tag = h2_tag.find("a")
            if a_tag:
                data_content_title = a_tag.get("data-content-title")
                data_content_titles.append(data_content_title)

        date_tags = soup.find_all("div", class_="col-xs-4 col-sm-2 tm-time")
        for date_tag in date_tags:
            time_tag = date_tag.find("time", class_="timestamp")
            if time_tag:
                timestamp_str = time_tag.get_text(strip=True)
                timestamp = parse_timestamp(timestamp_str)
                if timestamp:
                    timestamps.append(timestamp)
                    div_tag = date_tag.find_next("div", class_="timeline-content")
                    p_tag = div_tag.find("p")
                    if p_tag:
                        content_text = p_tag.get_text(strip=True)
                        content_texts.append(content_text)

    min_length = min(len(data_content_titles), len(timestamps), len(content_texts))
    data_content_titles = data_content_titles[:min_length]
    timestamps = timestamps[:min_length]
    content_texts = content_texts[:min_length]

    news = pd.DataFrame({
        "Title": data_content_titles,
        "Timestamp": timestamps,
        "Content": content_texts
    })

    news['Timestamp'] = pd.to_datetime(news['Timestamp'])
    news['Date'] = news['Timestamp'].dt.date
    news['Time'] = news['Timestamp'].dt.time
    news = news[['Title', 'Date', 'Time', 'Content']]

    news['Cleaned_Content'] = news['Content'].apply(clean_text)

    news['Summary'] = news['Cleaned_Content'].apply(generate_summary)

    summaries = news['Cleaned_Content']
    titles_data = []

    for summary in summaries:
        summary_keywords = keywords.keywords(summary).split()
        summary_words = summary.split()
        title = ' '.join(summary_words[:5]) + f" - {' '.join(summary_keywords)}"
        titles_data.append({'Generated_Title': title})

    titles_df = pd.DataFrame(titles_data)

    merged_df = pd.concat([news, titles_df], axis=1)

    sid = SentimentIntensityAnalyzer()
    sentiments = []

    for text in merged_df['Summary']:
        sentiment_scores = sid.polarity_scores(text)
        sentiments.append(sentiment_scores)

    merged_df['Sentiment'] = sentiments
    hot_threshold = 0.5
    merged_df['Hot'] = merged_df['Sentiment'].apply(lambda x: x['compound'] > hot_threshold)

    return merged_df

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text

def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, num_beams=4, min_length=30, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def parse_timestamp(timestamp_str):
    try:
        timestamp = dateutil.parser.parse(timestamp_str)
        timestamp = timestamp.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("Asia/Kolkata"))
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        if 'm' in timestamp_str:
            minutes_ago = int(timestamp_str.split('m')[0])
            return (datetime.now(pytz.timezone("Asia/Kolkata")) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")
        elif 'h' in timestamp_str:
            hours_ago = int(timestamp_str.split('h')[0])
            return (datetime.now(pytz.timezone("Asia/Kolkata")) - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
    return None

st.title('News article Summary App')
    # Add two buttons for text summarization and text classification
url = st.text_input('Enter URL:', 'https://www.thestar.com.my/news/latest/')
if st.button('Fetch News'):
    news_data = fetch_news_data(url)
    st.write("Fetched News Data:")
    st.write(news_data)

# Add two buttons for text summarization and text classification
if st.button('Text Summarization'):
    news_data = fetch_news_data(url)
    st.write("Text Summarization:")
    st.write(news_data['Summary'])

if st.button('Text Classification'):
    news_data = fetch_news_data(url)
    st.write("Text Classification:")
    st.write(news_data[['Sentiment', 'Hot']])