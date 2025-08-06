import torch
import numpy as np
from sklearn.cluster import KMeans
import umap
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked


class TeamClassifier:
    def __init__(self, model_name='google/siglip-base-patch16-224', device=None, n_clusters=2):
        self.device = 'cuda' if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.model_name = model_name
        self.n_clusters = n_clusters

        # Load SigLIP model and processor
        self.model = SiglipVisionModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # For clustering
        self.reducer = None
        self.kmeans = None
        self.player_team_dict = {}

    def _extract_embeddings(self, crops, batch_size=32):
        """
        Extract embeddings for a list of player crops (PIL Images).
        """
        embeddings_list = []
        self.model.eval()

        with torch.no_grad():
            for batch in chunked(crops, batch_size):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)

                # Use mean pooling across tokens
                batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                # Normalize embeddings (L2 norm)
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                embeddings_list.append(batch_embeddings.cpu().numpy())

                print(f"[INFO] Processing batch of {len(batch)} crops...")

        return np.concatenate(embeddings_list)

    def fit(self, crops):
        """
        Fit the model to the given player crops and cluster them into teams.
        """
        # Step 1: Extract embeddings
        print("[INFO] Extracting embeddings for player crops...")
        embeddings = self._extract_embeddings(crops)

        # Step 2: Dimensionality reduction with UMAP
        print("[INFO] Reducing dimensions with UMAP...")
        self.reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        reduced_embeddings = self.reducer.fit_transform(embeddings)

        # Step 3: Cluster into teams using KMeans
        print(f"[INFO] Clustering into {self.n_clusters} teams...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(reduced_embeddings)

        self.embeddings = embeddings
        self.reduced_embeddings = reduced_embeddings

        print("[INFO] Extracting embeddings for", len(crops), "crops...")

    def predict(self, crop):
        """
        Predict the team for a single player crop.
        """
        if self.kmeans is None or self.reducer is None:
            raise ValueError("You must call fit() before predict().")

        # Extract embedding for this crop
        embedding = self._extract_embeddings([crop])
        reduced_embedding = self.reducer.transform(embedding)

        # Predict cluster
        team_label = self.kmeans.predict(reduced_embedding)[0]
        return int(team_label) + 1  # Make it 1 or 2 instead of 0 or 1

    def assign_goalkeeper_by_proximity(self, players_dict, goalkeeper_dict):
        """
        Assign goalkeepers to teams based on distance to team centroids.
        players_dict: {id: {"bbox": [...], "team": 1 or 2}}
        goalkeeper_dict: {id: {"bbox": [...]}}

        Returns: {goalkeeper_id: team_id}
        """
        if len(players_dict) == 0 or len(goalkeeper_dict) == 0:
            return {}

        # Compute centroids for each team
        team_positions = {1: [], 2: []}
        for _, player in players_dict.items():
            bbox = player['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = bbox[3]  # bottom center
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
