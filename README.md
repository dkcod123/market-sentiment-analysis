# Market Sentiment Analysis & Opinion Mining Platform

## Prerequisites (for a brand new PC)

### 1. Python 3.10+
- Download and install from: https://www.python.org/downloads/
- During installation, check "Add Python to PATH".

### 2. Node.js (for Angular frontend)
- Download and install from: https://nodejs.org/en/download/
- Recommended: LTS version (16+)

### 3. Docker Desktop
- Download and install from: https://www.docker.com/products/docker-desktop/
- Required for containerization and deployment.

### 4. Git
- Download and install from: https://git-scm.com/downloads
- Useful for version control and cloning repositories.

### 5. (Optional) VS Code
- Download and install from: https://code.visualstudio.com/
- Recommended for development.

---

## Project Structure

```
Market Sentiment Analysis folder/
│
├── backend/      # FastAPI, Python, MongoDB, etc.
├── frontend/     # Angular 16+ dashboard
├── docker-compose.yml
└── README.md
```

---

## Setup Steps

### 1. Clone the repository (if using Git)
```
git clone <repo-url>
cd "Market Sentiment Analysis folder"
```

### 2. Backend Setup
- Install Python dependencies:
```
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup
- Install Angular CLI globally (if not already):
```
npm install -g @angular/cli@16
```
- Create or install frontend dependencies:
```
cd ../frontend
npm install
```

### 4. Docker Setup
- To run both backend and frontend using Docker:
```
docker-compose up --build
```

---

## Next Steps
- The backend and frontend folders will be populated in the following steps.
- Each step will be confirmed with you before proceeding. 

---

## API Endpoint Testing & Usage

### 1. Start the FastAPI Backend

If you haven’t already, install the backend dependencies:
```sh
cd backend
pip install -r requirements.txt
```

Then, start the FastAPI server (for local testing):
```sh
uvicorn main:app --reload
```
- This will run your API at: http://127.0.0.1:8000

---

### 2. Test the Endpoints (in order)

#### A. Fetch Twitter Data
Fetch and store tweets in MongoDB:
```
GET http://127.0.0.1:8000/fetch_twitter_data?limit=10
```

#### B. Fetch Reddit Data
Fetch and store Reddit posts/comments in MongoDB:
```
GET http://127.0.0.1:8000/fetch_reddit_data?limit=5
```
- Note: You need to set up your Reddit API credentials in your environment or in the code for this to work.

#### C. Clean the Data
Clean and preprocess all fetched data:
```
GET http://127.0.0.1:8000/clean_data
```

#### D. Run Sentiment Analysis
Classify all cleaned data with FinBERT:
```
GET http://127.0.0.1:8000/run_sentiment_analysis
```

---

### 3. Check MongoDB
- Use MongoDB Compass or another tool to view your collections:
  - `twitter_data`, `reddit_data` (raw)
  - `cleaned_twitter_data`, `cleaned_reddit_data` (cleaned + sentiment)

---

### 4. Troubleshooting
- If you get errors about missing packages, install them with pip.
- If you get errors about Reddit credentials, you’ll need to create a Reddit app and set the credentials in your environment or code. 