from sklearn.feature_extraction.text import TfidfVectorizer
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Fix for memory leak warning
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class ClusterAnalyzer:
    def __init__(self, n_clusters=5, use_distilbert=True):
        self.n_clusters = n_clusters
        self.use_distilbert = use_distilbert  # Flag to toggle between DistilBERT and processed content

    def analyze_clusters(self, raw_text_data=None,feature_data_np=None):
        
        metadata = {}

        if self.use_distilbert:
            if feature_data_np is None:
                raise ValueError("DistilBERT features are None but use_distilbert is set to True.")
            feature_data = feature_data_np
        else:
            if raw_text_data is None:
                raise ValueError("Raw text data is None but use_distilbert is set to False.")
            vectorizer = TfidfVectorizer()
            feature_data = vectorizer.fit_transform(raw_text_data).toarray()

        # Check if there are enough samples for clustering
        if feature_data.shape[0] < self.n_clusters:
            print(f"Warning: Number of samples ({feature_data.shape[0]}) is less than the number of clusters ({self.n_clusters}). Skipping clustering.")
            return {}, metadata

        # Standardize the features
        scaler = StandardScaler()
        print("Shape of the array:", feature_data.shape)  # Debugging
        print("Type of the array:", type(feature_data))  # Debugging
        print("First few elements:", feature_data[:2])  # Debugging

        if len(feature_data.shape) == 3:
            feature_data = feature_data.reshape(feature_data.shape[0], -1)

        try:
            scaled_array = scaler.fit_transform(feature_data)
        except ValueError as e:
            print(f"Error in scaling: {e}")

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)

        try:
            kmeans.fit(scaled_array)
        except ValueError as e:
            print(f"Error in clustering: {e}")
            return {}, metadata
    
        # Retrieve the cluster centers and labels
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
    
        # Organize the content into clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(feature_data[i])
    
        print("Type of clusters:", type(clusters))
        print("Value of clusters:", clusters)
    
        # Summary statistics for each cluster
        cluster_summaries = {}
        for label, content in clusters.items():
            cluster_summaries[label] = {
                "num_elements": len(content),
                "mean_feature": np.mean([feature_data[i] for i in range(feature_data.shape[0]) if cluster_labels[i] == label]),
                "diameter": np.max(pairwise_distances(content)),
                "radius": np.mean(pairwise_distances(content, [cluster_centers[label]]))
            }
    
        # Update the metadata dictionary
        metadata['clustering'] = {
            "cluster_centers": cluster_centers.tolist(),
            "cluster_labels": cluster_labels.tolist(),
            "cluster_summaries": cluster_summaries,
            "silhouette_score": silhouette_score(feature_data, cluster_labels),
            "inertia": kmeans.inertia_,
            "timestamp": datetime.now().isoformat()
        }
    
        # Return both cluster_results (clusters) and metadata
        return clusters, metadata


# Example usage
if __name__ == "__main__":
    analyzer = ClusterAnalyzer(n_clusters=3)
    sample_content = ["This is an example", "Another example", "Yet another example"]
    metadata = {"source_url": "https://www.example.com", "scraping_time": "2023-08-26 16:27:03", "text_length": 1234}
    updated_metadata = analyzer.analyze_clusters(sample_content, metadata)
    print("Updated Metadata:", updated_metadata)
