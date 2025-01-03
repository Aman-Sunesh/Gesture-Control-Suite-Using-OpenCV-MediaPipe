import cv2
import mediapipe as mp
import numpy as np
import pycaw
import time
import math
import sys
import screen_brightness_control as sbc
import HandTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from collections import deque


# Initialization
pTime = 0
camera_index = 0
vol = 0
width_cam = 700
height_cam = 500
menu_visible = True
current_mode = None
menu_timer_start = None
timer_duration = 5  # 3 seconds

# Set up video capture
cap = cv2.VideoCapture(camera_index)
cap.set(3, width_cam)
cap.set(4, height_cam)

if not cap.isOpened():
    print(f"Failed to open camera with index {camera_index}.")
    sys.exit()

detector = htm.handDetector(min_detection_confidence=0.75)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()  # Output: (-65.25, 0.0, 0.03125)
minVol = volume_range[0]
maxVol = volume_range[1]


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=15):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)

    if thickness > 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def display_menu(image):
    h, w, _ = image.shape

    main_menu_pt1 = (w - 240, 10)
    main_menu_pt2 = (w - 10, 210)

    draw_rounded_rect(image, main_menu_pt1, main_menu_pt2, (255, 255, 255, 120), -1, radius=20)
    draw_rounded_rect(image, main_menu_pt1, main_menu_pt2, (0, 0, 0), 2, radius=20)

    header_pt1 = (w - 240, 10)
    header_pt2 = (w - 10, 35)
    draw_rounded_rect(image, header_pt1, header_pt2, (200, 200, 200, 120), -1, radius=10)
    cv2.putText(image, "MENU", ((header_pt1[0] + header_pt2[0]) // 2 - 40, header_pt1[1] + 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

    menu_options = {
        "1. Volume Control": (header_pt1[0] + 10, 75),
        "2. Brightness Control": (header_pt1[0] + 10, 115),
        "3. Exit": (header_pt1[0] + 10, 155)
    }

    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, pos in menu_options.items():
        cv2.putText(image, text, pos, font, 0.6, (255, 255, 255), 2)


def draw_hand_landmarks_and_calculate_length(landmarks, image):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    cx = (thumb_tip[1] + index_tip[1]) // 2
    cy = (thumb_tip[2] + index_tip[2]) // 2

    cv2.circle(image, (thumb_tip[1], thumb_tip[2]), 15, (255, 0, 0), cv2.FILLED)
    cv2.circle(image, (index_tip[1], index_tip[2]), 15, (255, 0, 0), cv2.FILLED)
    cv2.circle(image, (cx, cy), 12, (255, 0, 255), cv2.FILLED)
    cv2.line(image, (thumb_tip[1], thumb_tip[2]), (index_tip[1], index_tip[2]), (255, 0, 0), 2, cv2.FILLED)

    length = math.hypot(index_tip[1] - thumb_tip[1], index_tip[2] - thumb_tip[2])

    if length < 35:
        cv2.circle(image, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

    return length

fingerCounts = deque(maxlen=10)


while True:
    ret, image = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break    
    
    image = cv2.flip(image, 1)

    # Detect hands in the image
    detector.findHands(image)
    results = detector.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = detector.findPosition(image, hand_landmarks, draw=False)

    # Count raised fingers
    finger_count = detector.count_fingers(image)

    # If in a mode and open palm is detected, return to menu
    if finger_count == 5 and current_mode is not None:
        menu_visible = True
        current_mode = None
        menu_timer_start = time.time()  # Start the timer

    if menu_visible:
        # Check if timer is active and not yet completed
        if menu_timer_start is not None:
            elapsed_time = time.time() - menu_timer_start
            if elapsed_time < timer_duration:
                # Display the countdown timer on the screen
                cv2.putText(image, f'Timer: {timer_duration - int(elapsed_time)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Reset the timer
                menu_timer_start = None

            # Keep displaying the menu without processing gestures
            display_menu(image)
        
        else:
            # Allow menu selection once the timer is finished
            display_menu(image)

            # Switch modes based on finger count
            if finger_count == 1:
                current_mode = "volume"
                menu_visible = False
            elif finger_count == 2:
                current_mode = "brightness"
                menu_visible = False
            elif finger_count == 3:
                break  # Exit the application 

    # Execute the mode function if not in menu
    if current_mode == "volume" and len(landmarks) != 0:
        length = draw_hand_landmarks_and_calculate_length(landmarks, image)

        # Volume control logic
        vol = np.interp(length, [30, 300], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

        current_volume_scalar = volume.GetMasterVolumeLevelScalar()
        device_volume = round(current_volume_scalar * 100, 2)
        

        # Define the volume bar and percentage display
        vol_per = device_volume
        vol_level = np.interp(device_volume, [0, 100], [400, 150])

        cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(image, (50, int(vol_level)), (85, 400), (0, 255, 0), cv2.FILLED)

        cv2.putText(image, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    elif current_mode == "brightness" and len(landmarks) != 0:
        length = draw_hand_landmarks_and_calculate_length(landmarks, image)

        # Brightness control logic
        brightness = np.interp(length, [30, 300], [0, 100])

        # Get current brightness for the first display only
        current_brightness = sbc.get_brightness(display=0)[0]  # Assuming there's at least one display

        # Set brightness only if there's a significant change
        if abs(current_brightness - brightness) > 1:  # Adjust threshold for sensitivity
            sbc.set_brightness(brightness, display=0)

        brightness_level = np.interp(brightness, [0, 100], [400, 150])

        cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(image, (50, int(brightness_level)), (85, 400), (0, 255, 0), cv2.FILLED)

        cv2.putText(image, f'{int(brightness)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, f'FPS: {int(fps)}', (10, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    h, w = image.shape[:2]
    cv2.putText(image, f'Fingers: {finger_count}', (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Gesture Control', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 