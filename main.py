from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamClassifier
import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
import supervision as sv
from PIL import Image
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


def main():
    video_frames = read_video('input_videos/mkd7.mp4')
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

    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(player_crops)

    # --- Dynamic team color extraction ---
    cluster_colors = {0: [], 1: []}

    for player_id, player_data in first_frame_players.items():
        crop = sv.crop_image(frame, player_data['bbox'])
        pil_crop = sv.cv2_to_pillow(crop)
        cluster_id = team_classifier.predict(pil_crop) - 1
        dom_color = get_dominant_color(crop)
        cluster_colors[cluster_id].append(dom_color)

    cluster_to_color = {}
    for cluster_id, colors in cluster_colors.items():
        if len(colors) > 0:
            avg_color = np.mean(colors, axis=0).astype(int)
            cluster_to_color[cluster_id] = avg_color
        else:
            cluster_to_color[cluster_id] = np.array([128, 128, 128])  # fallback gray

    print("[INFO] Cluster to team color mapping:")
    for cid, color in cluster_to_color.items():
        print(f" Cluster {cid}: {color}")

    print("[INFO] Assigning teams and colors across all frames...")

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

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    save_video(output_video_frames, 'output_videos/processed_video.avi')


if __name__ == "__main__":
    main()
