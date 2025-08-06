from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamClassifier  # NEW CLASS
import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
import supervision as sv
from PIL import Image


def main():
    # Read Video
    video_frames = read_video('input_videos/mkd7.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # Get object tracks (players, referees, goalkeepers, ball)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Interpolate missing ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # ===============================
    # ✅ Assign Players to Teams using SigLIP + UMAP + KMeans
    # ===============================
    print("[INFO] Extracting player crops for team classification...")
    first_frame_players = tracks['players'][0]

    # Collect player crops from the first frame
    player_crops = []
    player_ids = []
    frame = video_frames[0]

    for player_id, player_data in first_frame_players.items():
        bbox = player_data["bbox"]
        # Crop player from frame
        crop = sv.crop_image(frame, bbox)
        player_crops.append(sv.cv2_to_pillow(crop))  # Convert to PIL for SigLIP
        player_ids.append(player_id)

    # Initialize and fit TeamClassifier
    team_classifier = TeamClassifier(device="cpu")
    team_classifier.fit(player_crops)

    # Assign predicted team to all players across all frames
    print("[INFO] Assigning teams to players across frames...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # For simplicity, assign based on first frame clusters
            # Predict team using the first-frame reference
            bbox = track['bbox']
            crop = sv.crop_image(video_frames[frame_num], bbox)
            team = team_classifier.predict(sv.cv2_to_pillow(crop))

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = (0, 255, 0) if team == 1 else (0, 0, 255)

    # ===============================
    # ✅ Assign Goalkeepers by proximity (same logic as before)
    # ===============================
    for frame_num in range(len(tracks['goalkeepers'])):
        gk_assignments = team_classifier.assign_goalkeeper_by_proximity(
            tracks['players'][frame_num],
            tracks['goalkeepers'][frame_num]
        )

        for gk_id, team in gk_assignments.items():
            tracks['goalkeepers'][frame_num][gk_id]['team'] = team
            tracks['goalkeepers'][frame_num][gk_id]['team_color'] = (0, 255, 0) if team == 1 else (0, 0, 255)

    # ===============================
    # ✅ Assign Ball to Player or Goalkeeper
    # ===============================
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num in range(len(tracks['players'])):
        # Merge players and goalkeepers dictionaries for ball possession calculation
        merged_players = {}
        merged_players.update(tracks['players'][frame_num])
        merged_players.update(tracks['goalkeepers'][frame_num])

        # Ball detection for current frame
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(merged_players, ball_bbox)

        if assigned_player != -1:
            # Check if it's a player or goalkeeper
            if assigned_player in tracks['players'][frame_num]:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                tracks['goalkeepers'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['goalkeepers'][frame_num][assigned_player]['team'])
        else:
            # If no assignment, repeat last known team possession
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(-1)  # No team in control yet

    team_ball_control = np.array(team_ball_control)

    # ===============================
    # ✅ Draw output annotations
    # ===============================
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save processed video
    save_video(output_video_frames, 'output_videos/processed_video.avi')


if __name__ == "__main__":
    main()
