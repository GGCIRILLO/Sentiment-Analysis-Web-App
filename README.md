# Sentiment Analysis Web Application

This Flask web application allows users to express their mood with text and emojis, providing sentiment analysis results along with a suggested quote.

## Technologies Used

- **Flask**: Web framework for building the application.
- **FastAPI**: Used for handling HTTP exceptions.
- **Transformers Library**: Utilized to work with the sentiment analysis model.
- **HTML/CSS/JavaScript**: Front-end technologies for designing and styling.
- **Quote Library**: A custom library for generating quotes.
- **numpy**: Used for numerical operations in Python.

## Model

The sentiment analysis model used in this application is based on [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). It classifies input text into three categories: Negative, Neutral, and Positive.

## How the App Works

1. **Input**: Users enter text and emojis on the home page.

2. **Analysis**: The application preprocesses the input by appending emojis to the text. It then tokenizes the input and obtains sentiment scores using the pre-trained model.

3. **Sentiment Prediction**: The model predicts the sentiment label (Negative, Neutral, Positive) based on the obtained scores.

4. **Suggested Response**: A quote is generated based on the predicted sentiment. Positive sentiments trigger positive quotes, while other sentiments result in motivational quotes.

5. **Result Display**: The predicted sentiment, scores, and suggested response are displayed on a results page.


## Code Overview

- The Flask routes are defined in `app.py`.
- HTML templates are stored in the `templates` folder.
- Sentiment analysis is performed in the `analyze_sentiment` function.
- Quotes are generated based on sentiment in the `generate_suggestion` function.

Feel free to explore the code for more details!

## Running Locally

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask app with `python app.py`.
4. Access the application at [http://localhost:5000/](http://localhost:5000/).

## License

This project is licensed under the [MIT License](LICENSE).
