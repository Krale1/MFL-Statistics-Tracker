import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance=70):
        """
        max_player_ball_distance: max pixel distance to consider player controlling ball
        """
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players_dict, ball_bbox):
        """
        Assign the ball to the player closest to the ball within max distance.
        Returns the player_id assigned, or -1 if no player is close enough.
        """
        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players_dict.items():
            player_bbox = player['bbox']
            # Consider distances from left and right bottom corners of player bbox to ball center
            distance_left = measure_distance((player_bbox[0], player_bbox[3]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[3]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
