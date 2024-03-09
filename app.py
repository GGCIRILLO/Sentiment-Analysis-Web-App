from flask import Flask
from fastapi import HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
import random
from quote import quote
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

class SentimentRequest(BaseModel):
    text: str
    emojis: list

def preprocess(text, emojis):
    # Append emojis to the text
    processed_text = f"{text} {' '.join(emojis)}"
    return processed_text

def generate_suggestion(sentiment):
    if sentiment == "Positive":
        random_positive_word = random.choice(positive_words)
        return quote(random_positive_word, limit=1)
    else:
        random_motivational_word = random.choice(motivational_words)
        return quote(random_motivational_word, limit=1)

positive_words = [
    "inspiring", "uplifting", "joyful", "optimistic", "encouraging",
    "positive", "hopeful", "energetic", "empowering", "heartwarming",
    "upbeat", "radiant", "cheerful", "exuberant", "grateful",
    "content", "blissful", "ecstatic", "buoyant", "glorious",
    "vibrant", "spirited", "jovial", "merry", "euphoric",
    "triumphant", "festive", "elated", "serene", "buoyant",
    "whimsical", "exhilarated", "jubilant", "carefree", "lively",
    "refreshed", "hopeful", "sunny", "unstoppable", "happy",
    "heartening", "inspirited", "dynamic", "invigorated", "blissful",
    "sanguine", "gleeful", "resilient", "animated", "vivacious"
]

motivational_words = [
    "achieve", "persevere", "success", "believe", "dedication",
    "objective", "vision", "inspire", "ambition", "victory",
    "perseverance", "resilience", "determination", "tenacity", "persistence",
    "drive", "commitment", "endeavor", "devotion", "triumph",
    "accomplish", "resilient", "invigorated", "objective", "vision",
    "zeal", "purpose", "motivate", "hardwork", "determination",
    "commitment", "focus", "discipline", "persistence", "devotion",
    "tenacity", "effort", "endeavor", "inspire", "motivate",
    "perseverance", "success", "victory", "triumph", "accomplishment",
    "ambition", "objective", "motivation"
]

def analyze_sentiment(sentiment: SentimentRequest):
    try:
        # Preprocess and append emojis to the text
        processed_input = preprocess(sentiment.text, sentiment.emojis)
        
        # Tokenize input and obtain sentiment scores
        encoded_input = tokenizer(processed_input, return_tensors='pt', max_length=256, truncation=True)
        output = model(**encoded_input)
        scores = output.logits.softmax(dim=1).detach().numpy()[0]

        # Determine sentiment based on scores
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        predicted_sentiment = sentiment_labels[np.argmax(scores)]
        
        # Suggested response based on sentiment
        suggestion = generate_suggestion(predicted_sentiment)

        return {"predicted_sentiment": predicted_sentiment, "scores": scores.tolist(), "suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_route():
    text = request.form.get('text')
    emojis = request.form.get('emojis').split(',')

    # Create an instance of SentimentRequest
    sentiment_request = SentimentRequest(text=text, emojis=emojis)
    
    try:
        # Call analyze_sentiment with the SentimentRequest instance
        result = analyze_sentiment(sentiment_request)

        return render_template('result.html', result=result)
    except HTTPException as e:
        # Handle HTTPException, e.g., show an error page
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)