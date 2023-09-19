import pickle
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Load the trained model from the file
with open('logistic_regression_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the same vectorizer used during training
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.post('/predict_sentiment', response_model=dict)
async def predict_sentiment(data: dict):
    try:
        text = data['text']

        # Preprocess the user's input using the same vectorizer
        text_features = vectorizer.transform([text])

        # Make a prediction using the loaded model
        predicted_sentiment = classifier.predict(text_features)

        # Assuming that your model is binary (positive/negative sentiment)
        if predicted_sentiment[0] == 1:
            sentiment = "positive"
        else:
            sentiment = "negative"

        response = {
            "sentiment": sentiment
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
