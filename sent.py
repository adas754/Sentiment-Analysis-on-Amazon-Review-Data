import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model from the file
with open('logistic_regression_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the same vectorizer used during training
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    try:
        data = request.json
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

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
