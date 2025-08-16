import cv2
from ultralytics import YOLO

def run_video_detection(video_path=0):
    """
    Run YOLO object detection on a video.
    Args:
        video_path: Path to video file OR 0 for webcam
    """
    # Load YOLO model
    model = YOLO("yolo11n.pt")

    # Open video (0 = default webcam)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video stream")
            break

        # Run YOLO detection (on a single frame)
        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

        # Draw detections on the frame
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow("YOLO Video Detection", annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run with webcam (0) or change path to your video file
    run_video_detection(0)  # 👈 use 0 for webcam, or "video.mp4" for a file
