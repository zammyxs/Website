import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# UI Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
PURPLE = (255, 0, 255)
CYAN = (255, 255, 0)

# Exercise instructions
EXERCISE_INSTRUCTIONS = {
    "squat": "Stand with feet shoulder-width apart, then bend knees and lower hips",
    "arm_raise": "Raise arm straight up above shoulder level",
    "t_pose": "Extend both arms straight out to sides at shoulder height"
}

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def get_landmark_coords(landmarks, landmark, w, h):
    return [
        landmarks[landmark.value].x * w,
        landmarks[landmark.value].y * h
    ]

def detect_squat(knee_angle, hip_angle):
    if knee_angle > 120:
        return "Go lower, bend your knees more", RED
    elif knee_angle < 70:
        return "Don't go too low", YELLOW
    elif hip_angle < 60:
        return "Keep your back straighter", YELLOW
    return "Good form!", GREEN


def detect_arm_raise(shoulder, elbow, wrist):
    if wrist[1] < shoulder[1]:
        return "Arm raised correctly!", GREEN
    else:
        return "Raise your arm above your shoulder", YELLOW

def detect_t_pose(left_shoulder, right_shoulder, left_elbow, right_elbow):
    if (abs(left_shoulder[1] - left_elbow[1]) < 30 and
        abs(right_shoulder[1] - right_elbow[1]) < 30):
        return "Good T-pose!", GREEN
    else:
        return "Extend both arms horizontally for T-pose", YELLOW

def draw_text_with_background(img, text, position, font_scale, color, bg_color, thickness=1, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    x, y = position
    cv2.rectangle(img, (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + padding), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_control_panel(img, exercise, h):
    panel_height = 130  # Increased height to accommodate all text
    panel = np.zeros((panel_height, img.shape[1], 3), dtype=np.uint8)
    
    # Draw panel background with gradient
    for i in range(panel_height):
        alpha = i / panel_height
        color = tuple([int(50 * alpha + 20 * (1 - alpha))] * 3)
        cv2.line(panel, (0, i), (panel.shape[1], i), color, 1)
    
    # Add exercise title (shortened if needed)
    exercise_title = f"EXERCISE: {exercise.upper()}"
    cv2.putText(panel, exercise_title, (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
    
    # Add timestamp right-aligned
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(panel, timestamp, (panel.shape[1] - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    
    # Add instructions with proper wrapping
    instructions = EXERCISE_INSTRUCTIONS[exercise]
    if len(instructions) > 60:  # Wrap long instructions
        parts = [instructions[i:i+60] for i in range(0, len(instructions), 60)]
        for i, part in enumerate(parts):
            draw_text_with_background(panel, part, (20, 60 + i*25), 
                                   0.5, WHITE, BLACK)
    else:
        draw_text_with_background(panel, instructions, (20, 60), 
                               0.5, WHITE, BLACK)
    
    # Add controls with proper spacing
    controls = "1:Squat  2:Arm Raise  3:TPose  Q:Quit"
    cv2.putText(panel, controls, (20, panel_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
    
    # Combine panel with main image
    img[-panel_height:, :] = panel
    return img

# Key mapping for exercises
exercise_map = {
    ord('1'): "squat",
    ord('2'): "arm_raise",
    ord('3'): "t_pose"
}
exercise = "squat"  # Default
shoulder_history = []
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Draw control panel
    frame = draw_control_panel(frame, exercise, h)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Common joints
        left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        left_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, w, h)
        right_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
        left_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, w, h)
        right_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        left_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        left_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)

        feedback = ""
        feedback_color = WHITE
        if exercise == "squat":
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            draw_text_with_background(frame, f'Knee Angle: {int(knee_angle)}°', (50, 70), 
                                      0.7, YELLOW, BLACK)
            draw_text_with_background(frame, f'Hip Angle: {int(hip_angle)}°', (50, 110), 
                                      0.7, YELLOW, BLACK)
            feedback, feedback_color = detect_squat(knee_angle, hip_angle)


        elif exercise == "arm_raise":
            feedback, feedback_color = detect_arm_raise(left_shoulder, left_elbow, left_wrist)
            draw_text_with_background(frame, "Raise your left arm above your shoulder", (50, 70), 
                                     0.7, YELLOW, BLACK)

        elif exercise == "t_pose":
            feedback, feedback_color = detect_t_pose(left_shoulder, right_shoulder, left_elbow, right_elbow)
            draw_text_with_background(frame, "Extend both arms horizontally (T-pose)", (50, 70), 
                                     0.7, YELLOW, BLACK)

        if feedback:
            draw_text_with_background(frame, feedback, (w//2 - 200, 50), 
                                     0.8, feedback_color, BLACK)

        # Draw pose landmarks with custom styling
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=BLUE, thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=GREEN, thickness=2, circle_radius=2)
        )

    cv2.imshow('Workout Pose Corrector', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key in exercise_map:
        exercise = exercise_map[key]
        shoulder_history = []  # Reset history when changing exercises

cap.release()
cv2.destroyAllWindows()

def count_fingers(hand_landmarks):
    """Returns the number of raised fingers (1-5, 6 for all fingers + thumb)."""
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other 4 fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)