import cv2
import mediapipe as mp
import numpy as np
import time
from twilio.rest import Client
import uuid
from dotenv import load_dotenv
import os

load_dotenv() 

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
GUARDIAN_PHONE_NUMBER = os.getenv('GUARDIAN_PHONE_NUMBER')
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Fall detection parameters
FALL_DURATION_THRESHOLD = 5  # Seconds before sending notification
FALL_DISTANCE_THRESHOLD = 50  # Pixel distance for fall detection

def calculate_angle(a, b, c):
    """Calculate the angle between three points in degrees."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

def send_notification():
    """Send SMS and initiate a call when a fall persists."""
    print("Notification triggered: Fall detected for 5 seconds!")
    try:
        # Send SMS
        message = client.messages.create(
            body="🚨 Emergency! A fall has been detected for 5 seconds!",
            from_=TWILIO_PHONE_NUMBER,
            to=GUARDIAN_PHONE_NUMBER
        )
        print(f"SMS sent: {message.sid}")

        # Initiate call (replace URL with actual hosted XML)
        call = client.calls.create(
            url="http://yourserver.com/siren.xml",
            from_=TWILIO_PHONE_NUMBER,
            to=GUARDIAN_PHONE_NUMBER
        )
        print(f"Call initiated: {call.sid}")
    except Exception as e:
        print(f"Failed to send notification: {e}")

def get_landmark_coords(landmarks, landmark_type, image_width, image_height):
    """Extract pixel coordinates for a given landmark."""
    lm = landmarks[landmark_type]
    return int(lm.x * image_width), int(lm.y * image_height)

def visualize_point(image, point, label, color=(204, 252, 0), size=5, font_scale=0.5):
    """Draw a point and its label on the image."""
    x, y = point
    cv2.putText(image, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    cv2.circle(image, (x, y), size, color, -1)

def main():
    cap = cv2.VideoCapture("fall2.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fall_status = "No Fall Detected"
    stage = "standing"
    fall_start_time = None
    fall_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape

            # Skip if no landmarks detected
            if not results.pose_landmarks:
                print("No landmarks detected in this frame.")
                continue

            try:
                landmarks = results.pose_landmarks.landmark

                # Extract key landmark coordinates
                landmarks_dict = {
                    'nose': mp_pose.PoseLandmark.NOSE,
                    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
                    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
                    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
                    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
                    'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
                    'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
                    'left_heel': mp_pose.PoseLandmark.LEFT_HEEL,
                    'right_heel': mp_pose.PoseLandmark.RIGHT_HEEL,
                    'left_foot_index': mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                    'right_foot_index': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
                }

                coords = {key: get_landmark_coords(landmarks, lm, image_width, image_height)
                          for key, lm in landmarks_dict.items()}

                # Calculate midpoints
                midpoints = {
                    'left_arm_wrist_elbow': (
                        (coords['left_wrist'][0] + coords['left_elbow'][0]) // 2,
                        (coords['left_wrist'][1] + coords['left_elbow'][1]) // 2
                    ),
                    'right_arm_wrist_elbow': (
                        (coords['right_wrist'][0] + coords['right_elbow'][0]) // 2,
                        (coords['right_wrist'][1] + coords['right_elbow'][1]) // 2
                    ),
                    'left_arm_shoulder_elbow': (
                        (coords['left_shoulder'][0] + coords['left_elbow'][0]) // 2,
                        (coords['left_shoulder'][1] + coords['left_elbow'][1]) // 2
                    ),
                    'right_arm_shoulder_elbow': (
                        (coords['right_shoulder'][0] + coords['right_elbow'][0]) // 2,
                        (coords['right_shoulder'][1] + coords['right_elbow'][1]) // 2
                    ),
                    'body_shoulder_hip': (
                        (coords['left_shoulder'][0] + coords['right_shoulder'][0] + coords['left_hip'][0] + coords['right_hip'][0]) // 4,
                        (coords['left_shoulder'][1] + coords['right_shoulder'][1] + coords['left_hip'][1] + coords['right_hip'][1]) // 4
                    ),
                    'left_leg_hip_knee': (
                        (coords['left_hip'][0] + coords['left_knee'][0]) // 2,
                        (coords['left_hip'][1] + coords['left_knee'][1]) // 2
                    ),
                    'right_leg_hip_knee': (
                        (coords['right_hip'][0] + coords['right_knee'][0]) // 2,
                        (coords['right_hip'][1] + coords['right_knee'][1]) // 2
                    ),
                    'left_leg_knee_ankle': (
                        (coords['left_knee'][0] + coords['left_ankle'][0]) // 2,
                        (coords['left_knee'][1] + coords['left_ankle'][1]) // 2
                    ),
                    'right_leg_knee_ankle': (
                        (coords['right_knee'][0] + coords['right_ankle'][0]) // 2,
                        (coords['right_knee'][1] + coords['right_ankle'][1]) // 2
                    ),
                    'left_foot_index_heel': (
                        (coords['left_foot_index'][0] + coords['left_heel'][0]) // 2,
                        (coords['left_foot_index'][1] + coords['left_heel'][1]) // 2
                    ),
                    'right_foot_index_heel': (
                        (coords['right_foot_index'][0] + coords['right_heel'][0]) // 2,
                        (coords['right_foot_index'][1] + coords['right_heel'][1]) // 2
                    )
                }

                # Calculate body centers
                upper_body = (
                    sum([coords['nose'][0], midpoints['left_arm_wrist_elbow'][0], midpoints['right_arm_wrist_elbow'][0],
                         midpoints['left_arm_shoulder_elbow'][0], midpoints['right_arm_shoulder_elbow'][0], midpoints['body_shoulder_hip'][0]]) // 6,
                    sum([coords['nose'][1], midpoints['left_arm_wrist_elbow'][1], midpoints['right_arm_wrist_elbow'][1],
                         midpoints['left_arm_shoulder_elbow'][1], midpoints['right_arm_shoulder_elbow'][1], midpoints['body_shoulder_hip'][1]]) // 6
                )

                lower_body = (
                    sum([midpoints['left_leg_hip_knee'][0], midpoints['right_leg_hip_knee'][0], midpoints['left_leg_knee_ankle'][0],
                         midpoints['right_leg_knee_ankle'][0], midpoints['left_foot_index_heel'][0], midpoints['right_foot_index_heel'][0]]) // 6,
                    sum([midpoints['left_leg_hip_knee'][1], midpoints['right_leg_hip_knee'][1], midpoints['left_leg_knee_ankle'][1],
                         midpoints['right_leg_knee_ankle'][1], midpoints['left_foot_index_heel'][1], midpoints['right_foot_index_heel'][1]]) // 6
                )

                body_center = (
                    (upper_body[0] + lower_body[0]) // 2,
                    (upper_body[1] + lower_body[1]) // 2
                )

                # Calculate point of action (average of feet midpoints)
                point_of_action = (
                    (midpoints['left_foot_index_heel'][0] + midpoints['right_foot_index_heel'][0]) // 2,
                    (midpoints['left_foot_index_heel'][1] + midpoints['right_foot_index_heel'][1]) // 2
                )

                # Calculate angles
                angles = {
                    'elbow_l': calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    ),
                    'elbow_r': calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    ),
                    'shoulder_l': calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    ),
                    'shoulder_r': calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    ),
                    'hip_l': calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ),
                    'hip_r': calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ),
                    'knee_l': calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    ),
                    'knee_r': calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    )
                }

                # Visualize angles
                for key, angle in angles.items():
                    lm_key = {
                        'elbow_l': 'left_elbow', 'elbow_r': 'right_elbow',
                        'shoulder_l': 'left_shoulder', 'shoulder_r': 'right_shoulder',
                        'hip_l': 'left_hip', 'hip_r': 'right_hip',
                        'knee_l': 'left_knee', 'knee_r': 'right_knee'  # Fixed typo
                    }[key]
                    cv2.putText(image, str(angle), coords[lm_key], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Visualize points
                for key, point in coords.items():
                    visualize_point(image, point, key)
                for key, point in midpoints.items():
                    visualize_point(image, point, key)
                visualize_point(image, upper_body, 'upper_body', (277, 220, 0), 9)
                visualize_point(image, lower_body, 'lower_body', (277, 220, 0), 9)
                visualize_point(image, body_center, 'body', (0, 0, 255), 12)
                visualize_point(image, point_of_action, 'point_of_action', (0, 0, 255), 5)

                # Fall detection
                fall_distance = abs(point_of_action[0] - body_center[0])
                is_falling = fall_distance > FALL_DISTANCE_THRESHOLD
                is_standing = not is_falling

                if is_falling and stage != "falling":
                    stage = "falling"
                    print("Fall detected. Monitoring duration...")
                    fall_start_time = time.time()
                elif is_standing and stage == "falling":
                    stage = "standing"
                    print("Recovered from fall.")
                    fall_start_time = None

                # Check for prolonged fall
                if fall_start_time and time.time() - fall_start_time > FALL_DURATION_THRESHOLD:
                    send_notification()
                    fall_start_time = None
                    fall_count += 1

                # Visualize fall status
                if stage == "falling":
                    cv2.putText(image, 'fall', (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                # Display metrics
                cv2.putText(image, f"Distance: {fall_distance}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"Stage: {stage}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, f"Falls: {fall_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)

                # Render pose landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
                )

            except Exception as e:
                print(f"Processing error: {e}")
                continue

            # Display and handle exit
            cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()