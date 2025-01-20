from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import sklearn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Initialize the FastAPI app
app = FastAPI()
nltk.download('punkt_tab')
#Create vecotrizer   
train_data = pd.read_csv("./models/train.csv")
categories = list(train_data.columns[2:])
def embed(text):
    text_token = word_tokenize(text)
    
    eng_stopwords = stopwords.words('english')
    text_stop = [word for word in text_token if (word.lower() not in eng_stopwords) and word.isalpha()]
    
    stemmer = SnowballStemmer(language='english')
    text_stem = [stemmer.stem(word) for word in text_stop]
    return " ".join(text_stem)
# eng_stopwords = stopwords.words('english')
# max_features = 3000
# vectorizer = TfidfVectorizer(lowercase=True,stop_words=eng_stopwords,ngram_range=(1,2),max_features=max_features).fit(train_data.comment_text)
# Set up templates for HTML rendering
templates = Jinja2Templates(directory="templates")
vectorizer = joblib.load("./models/vectorizer3.pkl")
model = joblib.load("./models/model1.pkl")



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for input
class TextInput(BaseModel):
    text: str

# Mock toxicity analysis function

# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint to analyze toxicity
@app.post("/analyze")
async def analyze_text(input: TextInput):
    input_str = embed(input.text)
    text_vectorized = vectorizer.transform([input_str])
    
    # Predict probabilities for each category
    probabilities = model.predict_proba(text_vectorized)[0]
    print(probabilities)
    # Define categories
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create a result dictionary

    result = {category: round(prob * 100, 2) for category, prob in zip(categories, probabilities)}
    
    return {
        "text": input.text,
        "probabilities": result
    }   

# Serve static files for templates and assets