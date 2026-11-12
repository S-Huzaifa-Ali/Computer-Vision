import cv2
import mediapipe as mp
import math
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(point_a, point_b, point_c):
    """Return the angle ABC in degrees. Each point is an (x, y) pair."""
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c

    angle1 = math.atan2(ay - by, ax - bx)
    angle2 = math.atan2(cy - by, cx - bx)
    angle = abs(math.degrees(angle1 - angle2))
    return 360 - angle if angle > 180 else angle

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read webcam frame.")
            break

        start = time.perf_counter()

        # MediaPipe expects RGB images.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width, _ = frame.shape

            # Draw body landmarks and skeleton connections.
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            def pixel_point(landmark_id):
                landmark = landmarks[landmark_id]
                return (int(landmark.x * width), int(landmark.y * height))

            # Optional: calculate left elbow angle using shoulder, elbow, wrist.
            left_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            left_elbow = mp_pose.PoseLandmark.LEFT_ELBOW.value
            left_wrist = mp_pose.PoseLandmark.LEFT_WRIST.value

            # Display the angle only when all three landmarks are visible enough.
            ids = [left_shoulder, left_elbow, left_wrist]
            if all(landmarks[i].visibility > 0.5 for i in ids):
                angle = calculate_angle(
                    pixel_point(left_shoulder),
                    pixel_point(left_elbow),
                    pixel_point(left_wrist)
                )
                ex, ey = pixel_point(left_elbow)
                cv2.putText(
                    frame, f"Left elbow: {angle:.1f} deg",
                    (ex + 10, ey - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2
                )

            # Show normalized landmark coordinates for a few body joints.
            for name, landmark_id in [
                ("Nose", mp_pose.PoseLandmark.NOSE.value),
                ("Left shoulder", left_shoulder),
                ("Left wrist", left_wrist)
            ]:
                lm = landmarks[landmark_id]
                if lm.visibility > 0.5:
                    print(f"{name}: x={lm.x:.3f}, y={lm.y:.3f}")

        elapsed = time.perf_counter() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Real-Time Pose Estimation - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
