# app/api/api.py

from flask import Flask, request, jsonify
import pickle
import os

model_path = os.path.join('saved_model', 'model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

vectorizer = model['vectorizer']
kmeans = model['kmeans']
resolution_summary = model['summarized_resolutions']

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello from Flask API!"})

@app.route('/recommend', methods=['POST'])
def recommend_resolution():
    data = request.get_json()
    title = data.get('title')
    description = data.get('description')

    if not title or not description:
        return jsonify({'error': 'Both title and description are required.'}), 400

    text = title + ' ' + description
    vector = vectorizer.transform([text])
    cluster = kmeans.predict(vector)[0]

    resolution = resolution_summary.get(cluster, "No resolution found.")
    return jsonify({
        'predicted_cluster': int(cluster),
        'recommended_resolution': resolution
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
