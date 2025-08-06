import torch
import numpy as np
from sklearn.cluster import KMeans
import umap
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import cv2

def get_dominant_color(image, k=3):
    """
    Get dominant color of an image (PIL or numpy ndarray).
    Returns RGB tuple.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Convert to RGB if grayscale or with alpha channel
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Reshape image pixels to (num_pixels, 3)
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    # KMeans clustering on pixels
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)

    # Most frequent cluster center is dominant color
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant_color.astype(int)


class TeamClassifier:
    def __init__(self, model_name='google/siglip-base-patch16-224', device=None, n_clusters=2):
        self.device = 'cuda' if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.model_name = model_name
        self.n_clusters = n_clusters

        self.model = SiglipVisionModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)

        self.reducer = None
        self.kmeans = None
        self.cluster_to_color = {}  # Mapping cluster_id -> assigned RGB color

    def _extract_embeddings(self, crops, batch_size=32):
        embeddings_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in chunked(crops, batch_size):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)

                batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                embeddings_list.append(batch_embeddings.cpu().numpy())

        return np.concatenate(embeddings_list)

    def fit(self, crops):
        print("[INFO] Extracting embeddings for player crops...")
        embeddings = self._extract_embeddings(crops)

        print("[INFO] Reducing dimensions with UMAP...")
        self.reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        reduced_embeddings = self.reducer.fit_transform(embeddings)

        print(f"[INFO] Clustering into {self.n_clusters} teams...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(reduced_embeddings)

        self.embeddings = embeddings
        self.reduced_embeddings = reduced_embeddings

        # Define your team colors (in RGB)
        team_colors = {
            0: np.array([0, 255, 0]),   # green
            1: np.array([255, 0, 0])    # red, using bright red instead of blue for clarity
        }

        # Calculate average dominant color per cluster
        cluster_colors = {i: [] for i in range(self.n_clusters)}
        for crop, cluster_id in zip(crops, cluster_labels):
            dom_color = get_dominant_color(crop)
            cluster_colors[cluster_id].append(dom_color)

        avg_cluster_colors = {}
        for cluster_id, colors in cluster_colors.items():
            avg_color = np.mean(colors, axis=0)
            avg_cluster_colors[cluster_id] = avg_color

        # Match cluster average color to closest predefined team color
        self.cluster_to_color = {}
        for cluster_id, avg_color in avg_cluster_colors.items():
            distances = {team_id: np.linalg.norm(avg_color - col) for team_id, col in team_colors.items()}
            best_team = min(distances, key=distances.get)
            self.cluster_to_color[cluster_id] = team_colors[best_team]

        print("[INFO] Cluster to team color mapping:", self.cluster_to_color)

    def predict_batch(self, crops):
        if self.kmeans is None or self.reducer is None:
            raise ValueError("You must call fit() before predict_batch().")

        embeddings = self._extract_embeddings(crops)
        reduced_embeddings = self.reducer.transform(embeddings)
        cluster_labels = self.kmeans.predict(reduced_embeddings)

        # Map cluster label to team color RGB
        return [(int(cluster_id) + 1, self.cluster_to_color[cluster_id]) for cluster_id in cluster_labels]

    def assign_goalkeeper_by_proximity(self, players_dict, goalkeeper_dict):
        if len(players_dict) == 0 or len(goalkeeper_dict) == 0:
            return {}

        team_positions = {1: [], 2: []}
        for _, player in players_dict.items():
            bbox = player['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = bbox[3]
            team_positions[player['team']].append((center_x, center_y))

        if not team_positions[1] or not team_positions[2]:
            return {}

        team_1_centroid = np.mean(team_positions[1], axis=0)
        team_2_centroid = np.mean(team_positions[2], axis=0)

        gk_team_assignment = {}
        for gk_id, gk_data in goalkeeper_dict.items():
            gk_bbox = gk_data['bbox']
            gk_center = np.array([(gk_bbox[0] + gk_bbox[2]) / 2, gk_bbox[3]])

            dist_1 = np.linalg.norm(gk_center - team_1_centroid)
            dist_2 = np.linalg.norm(gk_center - team_2_centroid)

            gk_team_assignment[gk_id] = 1 if dist_1 < dist_2 else 2

        return gk_team_assignment
