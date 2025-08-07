from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamClassifier
import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
import supervision as sv
from PIL import Image
from sklearn.cluster import KMeans

# --- Improved dominant color extraction ---
def get_dominant_color(image, k=3):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Take only top 60% (focus on jersey area)
    height = image.shape[0]
    image = image[:int(height * 0.6), :, :]

    # Convert to HSV for grass filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Remove green pixels (grass)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    filtered = cv2.bitwise_and(image, image, mask=mask_inv)

    # Flatten and remove black pixels
    pixels = filtered.reshape(-1, 3)
    pixels = pixels[np.any(pixels > 0, axis=1)]

    # Fallback: if no pixels left, use original crop average
    if len(pixels) == 0:
        return np.mean(image.reshape(-1, 3), axis=0).astype(int)

    # Adjust k if too few pixels
    k = min(k, len(pixels))
    if k == 1:  # Only one pixel
        return pixels[0].astype(int)

    pixels = np.float32(pixels)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant_color.astype(int)


def main():
    video_frames = read_video('input_videos/Final.mp4')
    tracker = Tracker('models/best.pt')

    # Get tracks from YOLO + ByteTrack
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

    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(player_crops)

    # Compute cluster colors from first frame using dominant color
    cluster_colors = {0: [], 1: []}
    for player_id, player_data in first_frame_players.items():
        crop = sv.crop_image(frame, player_data['bbox'])
        cluster_id = team_classifier.predict(sv.cv2_to_pillow(crop)) - 1
        dom_color = get_dominant_color(crop)
        cluster_colors[cluster_id].append(dom_color)

    # Average color per cluster
    cluster_to_color = {}
    for cluster_id, colors in cluster_colors.items():
        if len(colors) > 0:
            avg_color = np.mean(colors, axis=0).astype(int)
            cluster_to_color[cluster_id] = avg_color
        else:
            cluster_to_color[cluster_id] = np.array([128, 128, 128])  # fallback gray

    # Compute color difference to check if teams look similar
    color_diff = np.linalg.norm(cluster_to_color[0] - cluster_to_color[1])
    print(f"[INFO] Color difference between teams: {color_diff}")
    if color_diff < 30:
        print("[WARNING] Team colors too similar! Switching to color-based clustering...")
        # Fallback: use color-based clustering on first frame crops
        all_colors = []
        for player_id, player_data in first_frame_players.items():
            crop = sv.crop_image(frame, player_data['bbox'])
            all_colors.append(get_dominant_color(crop))
        kmeans = KMeans(n_clusters=2, random_state=42).fit(all_colors)
        labels = kmeans.labels_
        cluster_colors = {0: [], 1: []}
        for idx, color in enumerate(all_colors):
            cluster_id = labels[idx]
            cluster_colors[cluster_id].append(color)
        for cluster_id, colors in cluster_colors.items():
            cluster_to_color[cluster_id] = np.mean(colors, axis=0).astype(int)

    print("[INFO] Final cluster to team color mapping:")
    for cid, color in cluster_to_color.items():
        print(f" Cluster {cid}: {color}")

    print("[INFO] Assigning teams and colors across all frames...")
    # Optimized: Use color matching for all frames
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            crop = sv.crop_image(video_frames[frame_num], track['bbox'])
            dom_color = get_dominant_color(crop)
            # Find closest cluster color
            distances = [np.linalg.norm(dom_color - c) for c in cluster_to_color.values()]
            cluster_id = int(np.argmin(distances))
            team_color = cluster_to_color.get(cluster_id, np.array([128, 128, 128]))
            tracks['players'][frame_num][player_id]['team'] = cluster_id + 1
            tracks['players'][frame_num][player_id]['team_color'] = tuple(team_color.tolist())

    # Assign goalkeepers by proximity using updated team info
    for frame_num in range(len(tracks['goalkeepers'])):
        gk_assignments = team_classifier.assign_goalkeeper_by_proximity(
            tracks['players'][frame_num],
            tracks['goalkeepers'][frame_num]
        )
        for gk_id, team in gk_assignments.items():
            team_color = cluster_to_color.get(team - 1, np.array([128, 128, 128]))
            tracks['goalkeepers'][frame_num][gk_id]['team'] = team
            tracks['goalkeepers'][frame_num][gk_id]['team_color'] = tuple(team_color.tolist())

    # Ball assignment
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

    # Draw annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    save_video(output_video_frames, 'output_videos/processed_video.avi')


if __name__ == "__main__":
    main()
