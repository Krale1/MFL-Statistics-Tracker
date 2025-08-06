import cv2

def read_video(video_path):
    """
    Reads a video file and returns a VideoCapture object.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        cv2.VideoCapture: VideoCapture object for the video file.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(frames, output_path):
    """
    Saves a list of frames as a video file.
    
    Args:
        frames (list): List of frames to save.
        output_path (str): Path to save the video file.
        fps (int): Frames per second for the output video.
    """
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()