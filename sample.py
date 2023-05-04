import pickle
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the saved model from the file
with open('decision_tree_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the vectorizer used to vectorize the training data
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Create a new Flask application
app = Flask(__name__)

# Define a route to handle requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the request
    new_text = request.json['text']

    # Vectorize the text data
    new_features = vectorizer.transform(new_text)

    # Make predictions on the text data
    new_pred_labels = clf.predict(new_features)

    # Sum the predictions to get the final label
    column_names = ["Depression", "Anxiety", "Stress"]
    column_sums = np.sum(new_pred_labels, axis=0)
    max_index = np.argmax(column_sums)
    final_label = column_names[max_index]

    # Return the predictions as a JSON object
    response = {
        'predictions': new_pred_labels.tolist(),
        'final_label': final_label
    }
    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(port=8010)
