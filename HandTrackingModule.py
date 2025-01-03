import cv2
import mediapipe as mp
from collections import deque

class handDetector():
    def __init__(self, 
                 static_image_mode=False, 
                 max_num_hands=2, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5,
                 draw_color=(255, 0, 255),
                 draw_size=2):
        
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.draw_color = draw_color
        self.draw_size = draw_size
        self.fingerCounts = deque(maxlen=10)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode, 
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence, 
            min_tracking_confidence=self.min_tracking_confidence)

    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        image = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=self.draw_color, thickness=self.draw_size),
                    self.mp_drawing.DrawingSpec(color=self.draw_color, thickness=self.draw_size))
        
        return image

    def findPosition(self, image, hand_landmarks, draw=True):
        landmarkList = []
        if hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape  # Get the dimensions of the image
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert landmark position to pixel coordinates
                landmarkList.append((id, cx, cy, lm.visibility))  # Store the landmark data
                if draw:
                    cv2.circle(image, (cx, cy), self.draw_size, self.draw_color, cv2.FILLED)
        return landmarkList

    def count_fingers(self, image):
        """
        Count the total extended fingers across all detected hands in the current frame.
        Uses a deque to store up to 10 recent finger counts and returns the average.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            # No hands detected: append 0 and return average of recent values
            self.fingerCounts.append(0)
            return sum(self.fingerCounts) // len(self.fingerCounts)

        totalFingers = 0
        # Pair up each hand landmark set with its handedness info
        hand_infos = zip(self.results.multi_hand_landmarks, self.results.multi_handedness)

        for hand_landmarks, hand_info in hand_infos:
            # Get pixel-coordinate landmarks
            landmarks = self.findPosition(image, hand_landmarks, draw=True)
            if not landmarks:
                continue

            # Determine if it's a right or left hand
            label = hand_info.classification[0].label
            is_right_hand = (label == 'Right')
            fingers = 0

            # Thumb logic: compare x-coordinates relative to the CMC (landmark #1)
            thumb_cmc_index = 1
            thumb_tip_index = 4
            # landmarks[i] = (i, cx, cy, visibility)
            # x-coordinate is landmarks[i][1]
            if (landmarks[thumb_tip_index][1] < landmarks[thumb_cmc_index][1] and is_right_hand):
                fingers += 1
            elif (landmarks[thumb_tip_index][1] > landmarks[thumb_cmc_index][1] and not is_right_hand):
                fingers += 1

            # Other fingers: compare y-coordinates of the tip with the PIP joint
            # tip IDs: [8, 12, 16, 20], PIP IDs: [6, 10, 14, 18]
            finger_tip_ids = [8, 12, 16, 20]
            finger_pip_ids = [6, 10, 14, 18]
            for tip_id, pip_id in zip(finger_tip_ids, finger_pip_ids):
                # y-coordinate is landmarks[i][2]
                if landmarks[tip_id][2] < landmarks[pip_id][2]:
                    fingers += 1

            # Add up fingers for this hand
            totalFingers += fingers

        # Store in deque and compute average
        self.fingerCounts.append(totalFingers)
        return sum(self.fingerCounts) // len(self.fingerCounts)