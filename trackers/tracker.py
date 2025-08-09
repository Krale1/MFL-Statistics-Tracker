from ultralytics import YOLO
import os
import pickle
import supervision as sv
import sys
import pandas as pd
import numpy as np
import cv2

sys.path.append('../')
from utils import get_center_of_bbox, get_width_of_bbox

REID_COLOR_THRESH = 40
REID_DIST_THRESH = 50
REID_MEMORY = 30

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.tracker.match_thresh = 0.7
        self.next_stable_id = 1
        self.recent_players = {}

    def _get_hsv(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0, 0, 0])
        crop = crop[int(crop.shape[0]*0.1):int(crop.shape[0]*0.5), int(crop.shape[1]*0.2):int(crop.shape[1]*0.8)]
        if crop.size == 0:
            return np.array([0, 0, 0])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv = hsv.reshape(-1, 3)
        return np.mean(hsv, axis=0)

    def _assign_stable_id(self, raw_id, bbox, hsv_color, frame_num):
        cx, cy = get_center_of_bbox(bbox)

        for prev_raw_id, info in self.recent_players.items():
            if frame_num - info["last_seen"] > REID_MEMORY:
                continue

            prev_cx, prev_cy = info["pos"]
            dist = np.linalg.norm([cx - prev_cx, cy - prev_cy])
            color_dist = np.linalg.norm(hsv_color - info["hsv"])

            if dist < REID_DIST_THRESH and color_dist < REID_COLOR_THRESH:
                self.recent_players[prev_raw_id]["last_seen"] = frame_num
                self.recent_players[prev_raw_id]["pos"] = (cx, cy)
                self.recent_players[prev_raw_id]["hsv"] = hsv_color
                return info["stable_id"]

        stable_id = self.next_stable_id
        self.next_stable_id += 1
        self.recent_players[raw_id] = {
            "stable_id": stable_id,
            "pos": (cx, cy),
            "hsv": hsv_color,
            "last_seen": frame_num
        }
        return stable_id

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.2)
            detections.extend(detections_batch)
        return detections
    
    #Probably the most important function in the project
    def get_object_tracks(self, frames, read_from_stub=True, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": [], "goalkeepers": []}

        for frame_num, (det, frame) in enumerate(zip(detections, frames)):
            cls_map = {v: k for k, v in det.names.items()}
            det_sv = sv.Detections.from_ultralytics(det)
            tracked = self.tracker.update_with_detections(det_sv)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["goalkeepers"].append({})

            for d in tracked:
                bbox = d[0].tolist()
                cls_id = d[3]
                raw_id = d[4]

                hsv = self._get_hsv(frame, bbox)

                if cls_id == cls_map.get('player'):
                    sid = self._assign_stable_id(raw_id, bbox, hsv, frame_num)
                    tracks["players"][frame_num][sid] = {"bbox": bbox}

                elif cls_id == cls_map.get('goalkeeper'):
                    sid = self._assign_stable_id(raw_id, bbox, hsv, frame_num)
                    tracks["goalkeepers"][frame_num][sid] = {"bbox": bbox}

                elif cls_id == cls_map.get('referee'):
                    tracks["referees"][frame_num][raw_id] = {"bbox": bbox}

            for d in det_sv:
                bbox = d[0].tolist()
                cls_id = d[3]
                
                if cls_id == cls_map.get('ball'):
                    tracks["ball"][frame_num][1] = {"bbox": bbox, "detected": True}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_triangle(self, frame, bbox, color):
        x, _ = get_center_of_bbox(bbox)
        y = int(bbox[1])
        triangle = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        color = tuple(int(c) for c in color)
        cv2.drawContours(frame, [triangle], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)
        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None, is_goalkeeper=False):
        color = tuple(int(c) for c in color)
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_width_of_bbox(bbox)
        thickness = 3 if is_goalkeeper else 2

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            w, h = 40, 20
            x1, y1 = x_center - w // 2, y2 + 15 - h // 2
            x2, y2b = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2b), color, cv2.FILLED)
            text_x = x1 + 12 - (10 if str(track_id).isdigit() and int(track_id) > 99 else 0)
            cv2.putText(frame, str(track_id), (text_x, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.65), int(h * 0.85)
        x2, y2 = int(w * 0.95), int(h * 0.97)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        history = team_ball_control[:frame_num + 1]
        t1 = (history == 1).sum()
        t2 = (history == 2).sum()
        total = t1 + t2
        pct1 = t1 / total if total else 0
        pct2 = t2 / total if total else 0

        cv2.putText(frame, f"Team 1 Possession: {pct1 * 100:.2f}%", (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.putText(frame, f"Team 2 Possession: {pct2 * 100:.2f}%", (x1 + 10, y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output = []
        for f, frame in enumerate(video_frames):
            frame = frame.copy()
            players = tracks["players"][f]
            goalkeepers = tracks["goalkeepers"][f]
            referees = tracks["referees"][f]
            balls = tracks["ball"][f]

            # Draw players
            for pid, p in players.items():
                frame = self.draw_ellipse(frame, p["bbox"], p.get("team_color", (255, 0, 0)), pid)
                if p.get("has_ball"): 
                    frame = self.draw_triangle(frame, p["bbox"], (0, 0, 255))

            # Draw goalkeepers
            for gid, gk in goalkeepers.items():
                frame = self.draw_ellipse(frame, gk["bbox"], gk.get("team_color", (0, 0, 255)), gid, is_goalkeeper=True)
                if gk.get("has_ball"): 
                    frame = self.draw_triangle(frame, gk["bbox"], (0, 0, 255))

            # Draw referees
            for rid, ref in referees.items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0, 255, 255))

            # 🟩 Always draw ball — from detection or interpolation
            # Ball logic
            if 1 in balls:
                # Detected or interpolated
                color = (0, 255, 0) if balls[1].get("detected", False) else (0, 180, 0)
                frame = self.draw_triangle(frame, balls[1]["bbox"], color)
            else:
                # No ball at all this frame → draw gray triangle at last known position
                if f > 0 and 1 in tracks["ball"][f - 1]:
                    last_bbox = tracks["ball"][f - 1][1]["bbox"]
                    frame = self.draw_triangle(frame, last_bbox, (128, 128, 128))

            # Draw team possession bar
            frame = self.draw_team_ball_control(frame, f, team_ball_control)

            output.append(frame)

        return output
