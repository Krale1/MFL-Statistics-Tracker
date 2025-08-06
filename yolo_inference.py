from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source='input_videos/08fd33_4.mp4',
    save=True,
    project='runs',     
    name='predict'         
)

print(results[0])
for box in results[0].boxes:
    print(box)
    print('---------------')