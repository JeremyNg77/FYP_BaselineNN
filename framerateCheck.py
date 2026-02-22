import cv2

video_path = 'Part 2.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Frame Rate: {fps} FPS")
print(f"Total Frames: {total_frames}")

cap.release()