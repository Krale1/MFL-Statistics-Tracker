from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamClassifier
import supervision as sv
from player_ball_assigner import PlayerBallAssigner
import numpy as np


def main():
    # Read video frames
    video_frames = read_video('input_videos/mkd7.mp4')

    # Initialize tracker
    tracker = Tracker('models/best.pt')

    # Get object tracks (players, referees, goalkeepers, ball)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Interpolate missing ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Extract player crops from the first frame for fitting
    print("[INFO] Extracting player crops for team classification...")
    first_frame_players = tracks['players'][0]

    player_crops = []
    for player_id, player_data in first_frame_players.items():
        bbox = player_data["bbox"]
        crop = sv.crop_image(video_frames[0], bbox)
        player_crops.append(sv.cv2_to_pillow(crop))

    # Initialize and fit TeamClassifier
    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(player_crops)

    # Assign teams to players across all frames using batch prediction per frame
    print("[INFO] Assigning teams to players across frames...")
    for frame_num, player_track in enumerate(tracks['players']):
        crops = []
        player_ids = []

        for player_id, track in player_track.items():
            bbox = track['bbox']
            crop = sv.crop_image(video_frames[frame_num], bbox)
            crops.append(sv.cv2_to_pillow(crop))
            player_ids.append(player_id)

        if len(crops) == 0:
            continue

        team_labels = team_classifier.predict_batch(crops)

        for pid, team in zip(player_ids, team_labels):
            tracks['players'][frame_num][pid]['team'] = team
            tracks['players'][frame_num][pid]['team_color'] = (0, 255, 0) if team == 1 else (0, 0, 255)

    # Assign goalkeepers by proximity
    for frame_num in range(len(tracks['goalkeepers'])):
        gk_assignments = team_classifier.assign_goalkeeper_by_proximity(
            tracks['players'][frame_num],
            tracks['goalkeepers'][frame_num]
        )
        for gk_id, team in gk_assignments.items():
            tracks['goalkeepers'][frame_num][gk_id]['team'] = team
            tracks['goalkeepers'][frame_num][gk_id]['team_color'] = (0, 255, 0) if team == 1 else (0, 0, 255)

    # Assign ball possession
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

    # Draw annotations and save video
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    save_video(output_video_frames, 'output_videos/processed_video.avi')


if __name__ == "__main__":
    main()
