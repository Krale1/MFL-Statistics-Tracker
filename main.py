from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamClassifier
import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
import supervision as sv
from sklearn.cluster import KMeans

# --- helper function to get dominant color ---
def get_dominant_color(image, k=3):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)

    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant_color.astype(int)

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def main():
    video_frames = read_video('input_videos/bul.mp4')
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    print("[INFO] Extracting player crops for team classification...")
    first_frame_players = tracks['players'][0]
    frame = video_frames[0]

    player_crops = []
    player_ids = []
    for player_id, player_data in first_frame_players.items():
        bbox = player_data["bbox"]
        crop = sv.crop_image(frame, bbox)
        player_crops.append(sv.cv2_to_pillow(crop))
        player_ids.append(player_id)

    # --- Step 1: Fit Team Classifier ---
    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(player_crops)

    # Assign cluster IDs based on embeddings
    cluster_assignments = [team_classifier.predict(crop) - 1 for crop in player_crops]

    # --- Step 2: Compute dominant colors per embedding cluster ---
    cluster_colors = {0: [], 1: []}
    for i, crop in enumerate(player_crops):
        dom_color = get_dominant_color(np.array(crop))
        cluster_colors[cluster_assignments[i]].append(dom_color)

    cluster_to_color = {
        cid: np.mean(colors, axis=0).astype(int) if len(colors) > 0 else np.array([128, 128, 128])
        for cid, colors in cluster_colors.items()
    }

    # --- Step 3: Validate color difference ---
    color_diff = color_distance(cluster_to_color[0], cluster_to_color[1])
    print(f"[INFO] Color difference between teams: {color_diff}")

    if color_diff < 50:  # Threshold for too similar
        print("[WARNING] Team colors too similar! Switching to color-based clustering...")
        
        # Use dominant colors for clustering
        dominant_colors = [get_dominant_color(np.array(crop)) for crop in player_crops]
        kmeans_color = KMeans(n_clusters=2, random_state=42).fit(dominant_colors)
        
        cluster_assignments = kmeans_color.labels_
        cluster_colors = {0: [], 1: []}
        for i, color in enumerate(dominant_colors):
            cluster_colors[cluster_assignments[i]].append(color)
        
        cluster_to_color = {
            cid: np.mean(colors, axis=0).astype(int) if len(colors) > 0 else np.array([128, 128, 128])
            for cid, colors in cluster_colors.items()
        }

    print("[INFO] Final cluster to team color mapping:")
    for cid, color in cluster_to_color.items():
        print(f" Cluster {cid}: {color}")

    print("[INFO] Assigning teams and colors across all frames...")

    # --- Assign teams and colors to all players ---
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            crop = sv.crop_image(video_frames[frame_num], track['bbox'])
            pil_crop = sv.cv2_to_pillow(crop)
            cluster_id = team_classifier.predict(pil_crop) - 1
            team_color = cluster_to_color.get(cluster_id, np.array([128, 128, 128]))

            tracks['players'][frame_num][player_id]['team'] = cluster_id + 1
            tracks['players'][frame_num][player_id]['team_color'] = tuple(team_color.tolist())

    # Assign Goalkeepers by proximity using cluster_id teams
    for frame_num in range(len(tracks['goalkeepers'])):
        gk_assignments = team_classifier.assign_goalkeeper_by_proximity(
            tracks['players'][frame_num],
            tracks['goalkeepers'][frame_num]
        )
        for gk_id, team in gk_assignments.items():
            team_color = cluster_to_color.get(team - 1, np.array([128, 128, 128]))
            tracks['goalkeepers'][frame_num][gk_id]['team'] = team
            tracks['goalkeepers'][frame_num][gk_id]['team_color'] = tuple(team_color.tolist())

    # --- Ball assignment ---
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num in range(len(tracks['players'])):
        merged_players = {}
        merged_players.update(tracks['players'][frame_num])
        merged_players.update(tracks['goalkeepers'][frame_num])

        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(merged_players, ball_bbox)

        if assigned_player != -1:
            if assigned_player in tracks['players'][frame_num]:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                tracks['goalkeepers'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['goalkeepers'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(-1)

    team_ball_control = np.array(team_ball_control)

    # --- Draw annotations and save video ---
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    save_video(output_video_frames, 'output_videos/processed_video.avi')

if __name__ == "__main__":
    main()
