from fastapi import FastAPI
from pymongo import MongoClient
import os
from typing import List
import snscrape.modules.twitter as sntwitter
from datetime import datetime
import praw
import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import timedelta
from fastapi import Query

app = FastAPI()

# Placeholder MongoDB URI (replace with your Atlas URI later)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["market_sentiment"]

twitter_collection = db["twitter_data"]
reddit_collection = db["reddit_data"]
cleaned_twitter_collection = db["cleaned_twitter_data"]
cleaned_reddit_collection = db["cleaned_reddit_data"]

# List of Indian stock keywords/hashtags/tickers
STOCK_KEYWORDS = [
    "#NSE", "#BSE", "#RELIANCE", "#TCS", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"
]
STOCK_REGEX = re.compile(r"\\b(" + "|".join([k.replace('#','') for k in STOCK_KEYWORDS]) + ")\\b", re.IGNORECASE)

# Placeholder Reddit API credentials (replace with your own for production)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "your_client_id")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "your_client_secret")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "market-sentiment-app")

# Download NLTK resources if not present
import nltk
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load FinBERT model and tokenizer
FINBERT_MODEL = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
label_map = {0: "neutral", 1: "positive", 2: "negative"}
score_map = {"positive": 1, "negative": -1, "neutral": 0}

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        label = label_map[label_id]
        score = score_map[label]
    return label, score, probs[0][label_id].item()

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove emojis and non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def extract_stock_symbols(text):
    return list(set([m.upper() for m in STOCK_REGEX.findall(text)]))

@app.get("/fetch_twitter_data")
def fetch_twitter_data(limit: int = 50):
    tweets = []
    for keyword in STOCK_KEYWORDS:
        for tweet in sntwitter.TwitterSearchScraper(f'{keyword} since:2023-01-01').get_items():
            if len(tweets) >= limit:
                break
            tweet_data = {
                "content": tweet.content,
                "author": tweet.user.username,
                "timestamp": tweet.date,
                "likes": tweet.likeCount,
                "retweets": tweet.retweetCount,
                "stock_mentions": [k for k in STOCK_KEYWORDS if k.replace('#','').upper() in tweet.content.upper()]
            }
            tweets.append(tweet_data)
        if len(tweets) >= limit:
            break
    if tweets:
        twitter_collection.insert_many(tweets)
    return {"fetched": len(tweets), "sample": tweets[:3]}

@app.get("/fetch_reddit_data")
def fetch_reddit_data(limit: int = 20):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    subreddits = ["IndianStockMarket", "StockMarketIndia", "nseindia"]
    posts = []
    for sub in subreddits:
        for submission in reddit.subreddit(sub).new(limit=limit):
            post_data = {
                "title": submission.title,
                "body": submission.selftext,
                "author": str(submission.author),
                "upvotes": submission.score,
                "timestamp": datetime.fromtimestamp(submission.created_utc),
                "stock_mentions": [k for k in STOCK_KEYWORDS if k.replace('#','').upper() in (submission.title + ' ' + submission.selftext).upper()],
                "comments": []
            }
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                comment_data = {
                    "body": comment.body,
                    "author": str(comment.author),
                    "upvotes": comment.score,
                    "timestamp": datetime.fromtimestamp(comment.created_utc),
                    "stock_mentions": [k for k in STOCK_KEYWORDS if k.replace('#','').upper() in comment.body.upper()]
                }
                post_data["comments"].append(comment_data)
            posts.append(post_data)
    if posts:
        reddit_collection.insert_many(posts)
    return {"fetched": len(posts), "sample": posts[:2]}

@app.get("/clean_data")
def clean_data():
    # Clean Twitter data with low-influence filter
    cleaned_twitter_collection.delete_many({})
    for doc in twitter_collection.find():
        # Filter: skip tweets with (likes + retweets) < 2
        if (doc.get("likes", 0) + doc.get("retweets", 0)) < 2:
            continue
        cleaned_doc = doc.copy()
        cleaned_doc["cleaned_content"] = clean_text(doc.get("content", ""))
        cleaned_doc["extracted_symbols"] = extract_stock_symbols(doc.get("content", ""))
        cleaned_twitter_collection.insert_one(cleaned_doc)
    # Clean Reddit data with low-influence filter
    cleaned_reddit_collection.delete_many({})
    for doc in reddit_collection.find():
        # Filter: skip posts with upvotes < 2
        if doc.get("upvotes", 0) < 2:
            continue
        cleaned_doc = doc.copy()
        cleaned_doc["cleaned_title"] = clean_text(doc.get("title", ""))
        cleaned_doc["cleaned_body"] = clean_text(doc.get("body", ""))
        cleaned_doc["extracted_symbols"] = extract_stock_symbols(doc.get("title", "") + " " + doc.get("body", ""))
        # Clean comments (no filter for now)
        cleaned_comments = []
        for comment in doc.get("comments", []):
            cleaned_comment = comment.copy()
            cleaned_comment["cleaned_body"] = clean_text(comment.get("body", ""))
            cleaned_comment["extracted_symbols"] = extract_stock_symbols(comment.get("body", ""))
            cleaned_comments.append(cleaned_comment)
        cleaned_doc["cleaned_comments"] = cleaned_comments
        cleaned_reddit_collection.insert_one(cleaned_doc)
    return {"message": "Data cleaned (low-influence filtered) and stored in cleaned_twitter_data and cleaned_reddit_data collections."}

@app.get("/run_sentiment_analysis")
def run_sentiment_analysis():
    # Twitter
    for doc in cleaned_twitter_collection.find():
        text = doc.get("cleaned_content", "")
        if not text.strip():
            continue
        label, score, confidence = classify_sentiment(text)
        cleaned_twitter_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"sentiment_label": label, "sentiment_score": score, "sentiment_confidence": confidence}}
        )
    # Reddit (title + body)
    for doc in cleaned_reddit_collection.find():
        text = (doc.get("cleaned_title", "") + " " + doc.get("cleaned_body", "")).strip()
        if not text:
            continue
        label, score, confidence = classify_sentiment(text)
        update_fields = {"sentiment_label": label, "sentiment_score": score, "sentiment_confidence": confidence}
        # Also classify comments
        cleaned_comments = doc.get("cleaned_comments", [])
        for i, comment in enumerate(cleaned_comments):
            ctext = comment.get("cleaned_body", "")
            if not ctext.strip():
                continue
            clabel, cscore, cconfidence = classify_sentiment(ctext)
            cleaned_comments[i]["sentiment_label"] = clabel
            cleaned_comments[i]["sentiment_score"] = cscore
            cleaned_comments[i]["sentiment_confidence"] = cconfidence
        update_fields["cleaned_comments"] = cleaned_comments
        cleaned_reddit_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": update_fields}
        )
    return {"message": "Sentiment analysis complete for all cleaned data."}

# Aggregation utility

def aggregate_sentiment(stock: str = None, timeframe: str = "24h"):
    now = datetime.utcnow()
    if timeframe.endswith("h"):
        hours = int(timeframe[:-1])
        start_time = now - timedelta(hours=hours)
    elif timeframe.endswith("d"):
        days = int(timeframe[:-1])
        start_time = now - timedelta(days=days)
    else:
        start_time = now - timedelta(hours=24)

    # Twitter aggregation
    twitter_query = {"timestamp": {"$gte": start_time}}
    if stock:
        twitter_query["extracted_symbols"] = stock.upper()
    twitter_docs = list(cleaned_twitter_collection.find(twitter_query))
    tw_scores, tw_weights = [], []
    for doc in twitter_docs:
        score = doc.get("sentiment_score", 0)
        weight = doc.get("likes", 0) + doc.get("retweets", 0)
        if weight == 0:
            weight = 1  # fallback to 1 if no engagement
        tw_scores.append(score * weight)
        tw_weights.append(weight)

    # Reddit aggregation
    reddit_query = {"timestamp": {"$gte": start_time}}
    if stock:
        reddit_query["extracted_symbols"] = stock.upper()
    reddit_docs = list(cleaned_reddit_collection.find(reddit_query))
    rd_scores, rd_weights = [], []
    for doc in reddit_docs:
        score = doc.get("sentiment_score", 0)
        weight = doc.get("upvotes", 0) + len(doc.get("cleaned_comments", []))
        if weight == 0:
            weight = 1
        rd_scores.append(score * weight)
        rd_weights.append(weight)
        # Include comments
        for comment in doc.get("cleaned_comments", []):
            cscore = comment.get("sentiment_score", 0)
            cweight = comment.get("upvotes", 0)
            if cweight == 0:
                cweight = 1
            rd_scores.append(cscore * cweight)
            rd_weights.append(cweight)

    total_score = sum(tw_scores) + sum(rd_scores)
    total_weight = sum(tw_weights) + sum(rd_weights)
    avg_sentiment = total_score / total_weight if total_weight else 0
    return {
        "stock": stock,
        "timeframe": timeframe,
        "total_weight": total_weight,
        "avg_sentiment": avg_sentiment,
        "twitter_count": len(twitter_docs),
        "reddit_count": len(reddit_docs)
    }

@app.get("/get_sentiment_score")
def get_sentiment_score(stock: str = Query(..., description="Stock symbol, e.g., TCS"), time: str = Query("24h", description="Time window, e.g., 24h or 7d")):
    result = aggregate_sentiment(stock, time)
    return result

@app.get("/all_stocks_sentiment")
def all_stocks_sentiment(time: str = Query("24h", description="Time window, e.g., 24h or 7d")):
    # Get all unique symbols from cleaned data
    twitter_symbols = cleaned_twitter_collection.distinct("extracted_symbols")
    reddit_symbols = cleaned_reddit_collection.distinct("extracted_symbols")
    all_symbols = set(twitter_symbols) | set(reddit_symbols)
    all_symbols = [s for s in all_symbols if s]
    results = [aggregate_sentiment(stock=s, timeframe=time) for s in all_symbols]
    return {"results": results} 