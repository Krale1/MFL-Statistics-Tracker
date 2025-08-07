# team_assigner.py

import numpy as np
from sklearn.cluster import KMeans
import cv2


class TeamClassifier:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_to_color = {}

    def _get_dominant_hsv(self, image):
        """
        Extract dominant HSV color from upper torso (centered).
        Returns mean HSV.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Crop center top 40% of the image
        h, w = image.shape[:2]
        crop = image[int(h * 0.05):int(h * 0.45), int(w * 0.2):int(w * 0.8)]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Mask out green grass
        lower_green = np.array([36, 50, 50])
        upper_green = np.array([86, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        hsv_masked = hsv[mask == 0]

        if len(hsv_masked) == 0:
            hsv_masked = hsv.reshape(-1, 3)

        return np.mean(hsv_masked, axis=0)

    def fit(self, player_crops):
        print("[INFO] Extracting dominant HSV from players...")
        self.hsv_colors = np.array([self._get_dominant_hsv(crop) for crop in player_crops])

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(self.hsv_colors)

        labels = self.kmeans.labels_
        cluster_colors = {i: [] for i in range(self.n_clusters)}

        for label, hsv in zip(labels, self.hsv_colors):
            cluster_colors[label].append(hsv)

        for i in range(self.n_clusters):
            mean_hsv = np.mean(cluster_colors[i], axis=0)
            bgr = cv2.cvtColor(np.uint8([[mean_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
            self.cluster_to_color[i] = bgr.astype(int)

        print("[INFO] Cluster color centers (BGR):", self.cluster_to_color)

    def predict(self, crop):
        hsv = self._get_dominant_hsv(crop)
        distances = [np.linalg.norm(hsv - center) for center in self.kmeans.cluster_centers_]
        cluster_id = int(np.argmin(distances))
        return cluster_id + 1

    def get_cluster_color_centroids(self):
        return self.cluster_to_color

    def assign_goalkeeper_by_proximity(self, players_dict, goalkeeper_dict):
        if len(players_dict) == 0 or len(goalkeeper_dict) == 0:
            return {}

        team_positions = {1: [], 2: []}
        for player in players_dict.values():
            bbox = player['bbox']
            x = (bbox[0] + bbox[2]) / 2
            y = bbox[3]
            team_positions[player['team']].append((x, y))

        gk_team_assignment = {}
        for gk_id, gk in goalkeeper_dict.items():
            bbox = gk['bbox']
            x = (bbox[0] + bbox[2]) / 2
            y = bbox[3]
            dists = []
            for t in [1, 2]:
                if team_positions[t]:
                    dists.append((t, np.linalg.norm(np.array([x, y]) - np.mean(team_positions[t], axis=0))))
            gk_team_assignment[gk_id] = min(dists, key=lambda x: x[1])[0]

        return gk_team_assignment
