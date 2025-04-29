import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import pickle
import os

def train_model(csv_path='data/incident_data_100.csv', n_clusters=15):
    # Load data
    incidents = pd.read_csv(csv_path)
    incidents['text'] = incidents['title'] + ' ' + incidents['description']

    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(incidents['text'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(x)
    incidents['cluster'] = clusters

    silhouette_avg = silhouette_score(x, clusters)
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    # Summarize resolutions
    cluster_resolutions = defaultdict(list)
    for _, row in incidents.iterrows():
        cluster_resolutions[row['cluster']].append(row['resolution'])

    summarized_resolutions = {
        cluster: Counter(res).most_common(1)[0][0]
        for cluster, res in cluster_resolutions.items()
    }

    model_bundle = {
        'vectorizer': vectorizer,
        'kmeans': kmeans,
        'summarized_resolutions': summarized_resolutions
    }

    os.makedirs('saved_model', exist_ok=True)
    with open('saved_model/model.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)

if __name__ == '__main__':
    train_model()
