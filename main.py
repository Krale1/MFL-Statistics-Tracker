# main.py

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamClassifier
from player_ball_assigner import PlayerBallAssigner

import cv2
import numpy as np
import supervision as sv


def main():
    video_frames = read_video('input_videos/Final.mp4')
    tracker = Tracker('models/newbest.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    print("[INFO] Extracting team info from first frame...")
    first_frame = video_frames[0]
    first_players = tracks['players'][0]
    player_crops = [sv.crop_image(first_frame, p["bbox"]) for p in first_players.values()]

    classifier = TeamClassifier()
    classifier.fit(player_crops)
    cluster_to_color = classifier.get_cluster_color_centroids()

    for frame_num, frame in enumerate(video_frames):
        for player_id, player in tracks['players'][frame_num].items():
            crop = sv.crop_image(frame, player["bbox"])
            team = classifier.predict(crop)
            color = cluster_to_color.get(team - 1, np.array([128, 128, 128]))
            player["team"] = team
            player["team_color"] = tuple(int(x) for x in color)

    for frame_num in range(len(tracks['goalkeepers'])):
        gk_assignments = classifier.assign_goalkeeper_by_proximity(
            tracks['players'][frame_num], tracks['goalkeepers'][frame_num]
        )
        for gk_id, team in gk_assignments.items():
            color = cluster_to_color.get(team - 1, np.array([128, 128, 128]))
            tracks['goalkeepers'][frame_num][gk_id]['team'] = team
            tracks['goalkeepers'][frame_num][gk_id]['team_color'] = tuple(color)

    print("[INFO] Tracking possession...")
    assigner = PlayerBallAssigner()
    team_ball_control = []
    last_team = -1
    smoothing_buffer = 5
    buffer = []

    for f in range(len(tracks['players'])):
        merged = {**tracks['players'][f], **tracks['goalkeepers'][f]}
        ball = tracks['ball'][f][1]['bbox']
        assigned = assigner.assign_ball_to_player(merged, ball)

        if assigned != -1:
            team = merged[assigned].get('team', -1)
            merged[assigned]['has_ball'] = True
        else:
            team = last_team

        buffer.append(team)
        if len(buffer) > smoothing_buffer:
            buffer.pop(0)

        smoothed_team = max(set(buffer), key=buffer.count) if buffer else -1
        team_ball_control.append(smoothed_team)
        last_team = smoothed_team

    team_ball_control = np.array(team_ball_control)
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    save_video(output_frames, 'output_videos/processed_video.avi')


if __name__ == "__main__":
    main()
