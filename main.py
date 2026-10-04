import os
import sys
import time
import warnings
from collections import Counter, deque

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Suppress background logging and deprecation warnings from TensorFlow and Protobuf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore')

import ctypes
from ctypes import wintypes
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Set PyAutoGUI pause to 0 for instantaneous, smooth cursor movement
pyautogui.PAUSE = 0
# Disable PyAutoGUI's automatic crash when touching corners;
# we provide our own boundary safety clamping and try-except handling instead.
pyautogui.FAILSAFE = False

# ==============================================================================
# CONFIGURATION & TUNING PARAMETERS
# ==============================================================================
# Window title constant for reference and window protection
WINDOW_TITLE = "AirTouch - Touchless Mouse Control"

# Cursor Movement Smoothing
# Higher values produce smoother cursor tracking at the expense of slight latency.
# Lower values make cursor response more immediate.
SMOOTH_FACTOR = 4

# Camera Interaction Frame Margins (in camera pixels)
# Prevents users from having to stretch their hands to extreme borders of the webcam view.
MARGIN_X = 90
MARGIN_Y = 70

# Rolling buffer size for majority-vote gesture classification (debouncing)
GESTURE_HISTORY_LEN = 7

# Pinch / Left Click Configuration
PINCH_COOLDOWN = 0.5           # Minimum seconds between left clicks
MIN_PINCH_CONFIRM = 2          # Consecutive frames required to confirm pinch before clicking
PINCH_DISTANCE_THRESHOLD = 0.35# Normalized distance threshold (thumb tip to index tip / palm size)
MIN_PINCH_RELEASE = 3          # Consecutive neutral frames required to confirm pinch release

# Drag & Drop Configuration (PINCH -> FIST)
DRAG_ARM_TIMEOUT = 2.0         # Max seconds between PINCH and FIST to initiate drag
MIN_FIST_CONFIRM = 2           # Consecutive FIST frames required to confirm drag initiation
MIN_FIST_RELEASE = 4           # Consecutive non-FIST frames required to end drag and drop

# Right Click (Two Fingers / Peace Sign) Configuration
RIGHT_CLICK_COOLDOWN = 0.6     # Cooldown in seconds to prevent rapid repeated right clicks
MIN_RIGHT_CLICK_CONFIRM = 4    # Consecutive frames required to confirm right click (~130ms)

# Double Click (Three Fingers) Configuration
DOUBLE_CLICK_COOLDOWN = 0.6    # Cooldown in seconds to prevent rapid repeated double clicks
MIN_DOUBLE_CLICK_CONFIRM = 4   # Consecutive frames required to confirm double click (~130ms)

# Browser Navigation (Four Fingers) Configuration
FOUR_FINGERS_COOLDOWN = 0.8    # Cooldown in seconds to prevent rapid repeated back navigation
MIN_FOUR_FINGERS_CONFIRM = 3   # Consecutive frames required to confirm browser back (~100ms)
MIN_FOUR_FINGERS_RELEASE = 3   # Consecutive neutral frames required to re-arm browser back

# Action Lock / Neutral Release Configuration
MIN_NEUTRAL_CONFIRM = 3        # Consecutive frames in neutral gesture required to unlock action
ACTION_LOCK_MIN_DURATION = 0.35# Minimum lock duration in seconds before neutral release is accepted

# Touchless Scrolling Configuration
SCROLL_TRIGGER_DISTANCE = 50   # Vertical pixel distance relative to entry position to scroll
SCROLL_DEADZONE = 20           # Deadzone for minor hand tremor without resetting baseline
SCROLL_COOLDOWN = 0.4          # Minimum cooldown in seconds between scroll pulses
SCROLL_CLICKS = 150            # Scroll unit amount for pyautogui.scroll (positive=up, negative=down)
DIRECTION_REVERSAL_DELAY = 0.65# Delay required before opposite scroll direction can trigger

# Action Feedback HUD Display Duration (seconds)
ACTION_FEEDBACK_DURATION = 0.8 # Duration for action notifications to remain active on the HUD

# Visual Color Palettes (BGR format for OpenCV)
GESTURE_COLORS = {
    "POINT": (0, 230, 255),        # Vibrant Yellow-Cyan
    "PINCH": (255, 105, 180),      # Hot Pink
    "RIGHT CLICK": (255, 80, 255), # Magenta
    "DOUBLE CLICK": (255, 230, 0), # Bright Cyan
    "FOUR FINGERS": (255, 180, 0), # Vibrant Cyan-Orange
    "OPEN PALM": (60, 230, 80),    # Bright Green
    "FIST": (50, 130, 255),        # Vibrant Orange
    "WAITING...": (160, 160, 160), # Light Gray
    "NO HAND": (110, 110, 110)     # Muted Gray
}

STATUS_COLORS = {
    "ACTIVE": (60, 230, 80),       # Green
    "PAUSED": (0, 215, 255),       # Amber / Yellow
    "DRAGGING": (0, 165, 255),     # Deep Orange
    "ARMED": (0, 200, 255),        # Light Orange
    "BROWSER BACK": (255, 180, 0), # Cyan-Orange
    "BLOCKED": (60, 80, 240),      # Crimson / Red
    "IDLE": (160, 160, 160),       # Gray
}

ACTION_COLORS = {
    "LEFT CLICK": (255, 105, 180), # Pink
    "RIGHT CLICK": (255, 80, 255), # Magenta
    "DOUBLE CLICK": (255, 230, 0), # Cyan
    "BROWSER BACK": (255, 180, 0), # Cyan-Orange
    "DRAG START": (0, 165, 255),   # Orange
    "DRAG END": (60, 230, 80),     # Green
    "SCROLL UP": (0, 240, 255),    # Yellow
    "SCROLL DOWN": (0, 240, 255),  # Yellow
    "NONE": (140, 140, 140)        # Muted Gray
}


def draw_glass_panel(img, pt1, pt2, bg_color=(20, 20, 20), border_color=(65, 65, 65), alpha=0.75):
    """
    Draw a semi-transparent dark HUD panel with a subtle border.
    Allows hand landmarks and camera feed underneath to remain partially visible.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)


def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between two 2D points (x, y).
    Works with MediaPipe landmark objects or (x, y) tuples.
    """
    x1, y1 = (p1.x, p1.y) if hasattr(p1, 'x') else p1
    x2, y2 = (p2.x, p2.y) if hasattr(p2, 'x') else p2
    return np.hypot(x1 - x2, y1 - y2)


def is_finger_extended(landmarks, tip_id, pip_id, wrist_id=0):
    """
    Determine if a finger is open (extended) or closed (folded).
    A finger is considered extended if its tip is farther from the wrist
    than its intermediate joint (PIP). This remains reliable even when
    the hand is tilted.
    """
    wrist = landmarks[wrist_id]
    tip = landmarks[tip_id]
    pip = landmarks[pip_id]

    tip_dist = calculate_distance(tip, wrist)
    pip_dist = calculate_distance(pip, wrist)
    return tip_dist > pip_dist


def detect_gesture(landmarks):
    """
    Recognize hand gesture based on 21 MediaPipe landmarks.
    
    Gestures detected:
    1. PINCH: Thumb tip and Index finger tip are brought close together.
    2. POINT: Index finger is extended while other fingers are folded.
    3. OPEN PALM: All four fingers (and thumb) are open/extended.
    4. FIST: All four fingers are closed/folded.
    """
    # Reference distance: distance between wrist (0) and middle finger MCP (9).
    # Normalizes scale so distance checks work regardless of hand distance from camera.
    palm_size = calculate_distance(landmarks[0], landmarks[9])
    if palm_size == 0:
        return "UNKNOWN"

    # Check extension status for the four main fingers
    index_open = is_finger_extended(landmarks, tip_id=8, pip_id=6)
    middle_open = is_finger_extended(landmarks, tip_id=12, pip_id=10)
    ring_open = is_finger_extended(landmarks, tip_id=16, pip_id=14)
    pinky_open = is_finger_extended(landmarks, tip_id=20, pip_id=18)

    # Check thumb extension relative to pinky base (17)
    thumb_to_pinky_base = calculate_distance(landmarks[4], landmarks[17])
    thumb_mcp_to_pinky_base = calculate_distance(landmarks[2], landmarks[17])
    thumb_open = thumb_to_pinky_base > thumb_mcp_to_pinky_base

    # Distance between thumb tip (4) and index tip (8)
    pinch_dist = calculate_distance(landmarks[4], landmarks[8])
    normalized_pinch_dist = pinch_dist / palm_size

    # Total open fingers among index, middle, ring, pinky
    fingers_open_count = sum([index_open, middle_open, ring_open, pinky_open])

    # 1. FIST Gesture:
    # All 4 fingers are folded into the palm.
    if fingers_open_count == 0:
        return "FIST"

    # 2. OPEN PALM Gesture (Highest open-finger priority):
    # All 4 fingers are open and thumb is extended
    if fingers_open_count == 4 and thumb_open:
        return "OPEN PALM"

    # 3. FOUR FINGERS Gesture (Browser Back Navigation):
    # All 4 main fingers extended, thumb folded/tucked across palm
    if fingers_open_count == 4 and not thumb_open:
        return "FOUR FINGERS"

    # 4. DOUBLE CLICK Gesture (Three Fingers: Index + Middle + Ring extended):
    # Evaluated before two-finger and pinch gestures to prevent sub-gesture race conditions
    if index_open and middle_open and ring_open and not pinky_open:
        return "DOUBLE CLICK"

    # 4. RIGHT CLICK Gesture (Two Fingers / Peace Sign: Index + Middle extended):
    # Evaluated before one-finger and pinch gestures
    if index_open and middle_open and not ring_open and not pinky_open:
        return "RIGHT CLICK"

    # 5. PINCH Gesture:
    # Thumb tip and Index tip are brought together (< 35% of palm size)
    # Strictly requires middle, ring, and pinky to be folded (cannot overlap with 2 or 3 fingers)
    if normalized_pinch_dist < PINCH_DISTANCE_THRESHOLD and not middle_open and not ring_open and not pinky_open:
        return "PINCH"

    # 6. POINT Gesture:
    # Index finger extended, while other 3 fingers are curled in
    if index_open and not middle_open and not ring_open and not pinky_open:
        return "POINT"

    return "WAITING..."


def is_cursor_over_app_window(cursor_x=None, cursor_y=None, window_name=WINDOW_TITLE):
    """
    Check if the cursor is hovering over the AirTouch camera window or its title bar.
    This prevents accidental interaction (clicking or scrolling) targeting the camera window.
    """
    if cursor_x is None or cursor_y is None:
        try:
            pos = pyautogui.position()
            cursor_x, cursor_y = pos.x, pos.y
        except Exception:
            return False
    try:
        rect = cv2.getWindowImageRect(window_name)
        if rect is not None and len(rect) == 4:
            wx, wy, ww, wh = rect
            # Include title bar (~40px above client area) and border margins
            if (wx - 10 <= cursor_x <= wx + ww + 10) and (wy - 45 <= cursor_y <= wy + wh + 10):
                return True
    except Exception:
        pass
    return False


def main():
    # 1. Initialize MediaPipe Hands solution and drawing utilities
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 2. Configure Hand Tracking
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # 3. Get monitor screen dimensions for coordinate mapping
    try:
        screen_w, screen_h = pyautogui.size()
    except Exception:
        screen_w, screen_h = 1920, 1080

    # 4. Open webcam (0 is default camera with DirectShow backend for Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*'MJPG')
    )

    # Warm up camera before entering the main loop
    warmup_success = False
    for i in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            warmup_success = True
            break
        time.sleep(0.1)

    if not warmup_success:
        print("Error: Webcam opened but failed to provide frames.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Pre-create named window and pin it as always-on-top
    # This prevents the window from being hidden/sent to background when clicking other apps
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    print("AirTouch Touchless Mouse Control running...")
    print(f"Screen Resolution: {screen_w}x{screen_h}")
    print("Gestures:")
    print(" - POINT       : Move Mouse Cursor & Vertical Scroll")
    print(" - PINCH       : Single Left Click (Safely blocked over camera window)")
    print(" - RIGHT CLICK : Two Fingers / Peace Sign (Single Right Click)")
    print(" - DOUBLE CLICK: Three Fingers (Index + Middle + Ring extended)")
    print(" - FOUR FINGERS: Browser Back (Navigate to previous page)")
    print(" - PINCH+FIST  : Drag and Drop (PINCH then FIST to drag, release to drop)")
    print(" - OPEN PALM   : Pause Control")
    print(" - FIST        : Idle / No Action (without preceding pinch)")
    print("Press 'q' to quit.")

    # Rolling buffer to stabilize gesture classification (debouncing)
    gesture_history = deque(maxlen=GESTURE_HISTORY_LEN)

    # Mouse smoothing variables
    prev_mouse_x, prev_mouse_y = None, None
    smooth_factor = SMOOTH_FACTOR

    # Interaction box margin in camera pixels
    margin_x = MARGIN_X
    margin_y = MARGIN_Y

    # Single Source of Truth: app_state machine
    # States: "NORMAL", "DRAG_ARMED", "DRAGGING"
    app_state = "NORMAL"
    pinch_released = True          # True when hand is not pinching (ready for next physical pinch)
    pinch_cooldown_until = 0.0     # Cooldown timestamp for pinch left clicks
    pinch_confirm_frames = 0       # Consecutive frames required to confirm pinch
    pinch_release_frames = 0       # Consecutive neutral frames required to confirm pinch release
    drag_armed_time = 0.0          # Timestamp when PINCH armed the drag
    fist_confirm_frames = 0        # Consecutive FIST frames to confirm drag start
    fist_release_frames = 0        # Consecutive non-FIST frames to confirm ending drag

    # Gesture Action Lock & Priority State Machine
    action_lock = None             # None, "DOUBLE_CLICK", "RIGHT_CLICK", "PINCH", "DRAGGING"
    action_lock_time = 0.0
    neutral_confirm_frames = 0     # Consecutive frames in neutral gesture required to unlock

    # Right Click variables (Debounce + Cooldown + Confirmation)
    right_click_released = True    # True when hand is not in RIGHT CLICK gesture
    right_click_cooldown_until = 0.0
    right_click_confirm_frames = 0 # Consecutive frames required to confirm right click

    # Double Click variables (Debounce + Cooldown + Confirmation)
    double_click_released = True   # True when hand is not in DOUBLE CLICK gesture
    double_click_cooldown_until = 0.0
    double_click_confirm_frames = 0# Consecutive frames required to confirm double click

    # Browser Navigation variables (Debounce + Cooldown + Confirmation)
    four_fingers_released = True   # True when hand is not in FOUR FINGERS gesture
    four_fingers_cooldown_until = 0.0
    four_fingers_confirm_frames = 0# Consecutive frames required to confirm four fingers
    four_fingers_release_frames = 0# Consecutive neutral frames required to confirm release

    # Scrolling variables (Mouse wheel-based via pyautogui.scroll)
    scroll_start_y = None          # Starting vertical position of index tip when entering POINT
    scroll_cooldown_until = 0.0    # Monotonic timestamp until next scroll action allowed
    last_scroll_time = 0.0         # Timestamp of last scroll action
    last_scroll_direction = None   # Direction label for HUD display
    last_scroll_dir = None         # Direction of last scroll ("UP" or "DOWN") for recovery suppression
    last_scroll_dir_time = 0.0     # Timestamp when direction was triggered
    scroll_deadzone_frames = 0     # Consecutive frames in deadzone before direction resets

    # Action feedback HUD tracking
    hud_action = "NONE"
    hud_action_expiry = 0.0

    # Consecutive frame failure counters
    frame_failures = 0
    MAX_FRAME_FAILURES = 30

    try:
        while True:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_failures += 1
                print(f"Warning: Failed to grab webcam frame ({frame_failures}/{MAX_FRAME_FAILURES})")

                if frame_failures >= MAX_FRAME_FAILURES:
                    print("Error: Webcam frame capture failed repeatedly.")
                    break

                time.sleep(0.05)
                continue

            frame_failures = 0

            # Flip horizontally for a natural mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # MediaPipe requires RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            raw_gesture = "NO HAND"
            index_tip_pos = None
            hand_drag_pos = None

            # Draw the active interaction bounding box (guideline for cursor control)
            cv2.rectangle(
                frame,
                (margin_x, margin_y),
                (w - margin_x, h - margin_y),
                (70, 70, 70),
                1,
                cv2.LINE_AA
            )

            # If a hand is detected:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 1. Draw 21 landmarks and connection lines
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 2. Draw landmark ID numbers
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.putText(
                            frame,
                            str(idx),
                            (cx + 5, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                    # 3. Detect current gesture
                    raw_gesture = detect_gesture(hand_landmarks.landmark)

                    # Get Index Finger Tip (Landmark 8) pixel coordinates
                    tip = hand_landmarks.landmark[8]
                    tip_px_x = int(tip.x * w)
                    tip_px_y = int(tip.y * h)
                    index_tip_pos = (tip_px_x, tip_px_y)

                    # Get Middle Finger MCP (Landmark 9) for stable drag tracking during FIST
                    mcp9 = hand_landmarks.landmark[9]
                    hand_drag_pos = (int(mcp9.x * w), int(mcp9.y * h))

                    # Landmark processing complete
            else:
                gesture_history.clear()
                prev_mouse_x, prev_mouse_y = None, None
                scroll_start_y = None
                scroll_cooldown_until = 0.0
                last_scroll_direction = None
                last_scroll_dir = None
                fist_confirm_frames = 0
                pinch_confirm_frames = 0
                pinch_release_frames = 0
                right_click_confirm_frames = 0
                double_click_confirm_frames = 0
                four_fingers_confirm_frames = 0
                four_fingers_release_frames = 0
                neutral_confirm_frames = 0
                action_lock = None
                pinch_released = True
                right_click_released = True
                double_click_released = True
                four_fingers_released = True

                # Handle no-hand situation for active drag
                if app_state == "DRAGGING":
                    fist_release_frames += 1
                    if fist_release_frames >= MIN_FIST_RELEASE:
                        try:
                            pyautogui.mouseUp(button='left')
                            print("Action: Drag ended.")
                            hud_action = "DRAG END"
                            hud_action_expiry = time.time() + ACTION_FEEDBACK_DURATION
                        except Exception:
                            pass
                        app_state = "NORMAL"
                        fist_release_frames = 0
                elif app_state == "DRAG_ARMED":
                    app_state = "NORMAL"

            # Add raw gesture to history buffer for stabilization
            gesture_history.append(raw_gesture)

            # Get stabilized gesture via majority vote
            if gesture_history:
                stable_gesture = Counter(gesture_history).most_common(1)[0][0]
            else:
                stable_gesture = "NO HAND"

            # --- Single Source of Truth State Machine ---
            control_status = "IDLE"
            current_time = time.time()

            try:
                # ==============================================================
                # STATE 1: NORMAL
                # ==============================================================
                if app_state == "NORMAL":
                    # Neutral state tracking for unlocking action lock
                    # Unlocks action_lock when hand returns to neutral for MIN_NEUTRAL_CONFIRM frames
                    if stable_gesture in ("POINT", "OPEN PALM", "FIST", "NO HAND"):
                        neutral_confirm_frames += 1
                        if neutral_confirm_frames >= MIN_NEUTRAL_CONFIRM:
                            if action_lock in ("DOUBLE_CLICK", "RIGHT_CLICK", "PINCH", "FOUR_FINGERS"):
                                if current_time >= action_lock_time + 0.35:
                                    action_lock = None
                    else:
                        neutral_confirm_frames = 0

                    # Track physical release of pinch to arm for next click
                    # Requires confirmed open/neutral posture (POINT or OPEN PALM) for consecutive frames
                    if stable_gesture in ("POINT", "OPEN PALM"):
                        pinch_confirm_frames = 0
                        pinch_release_frames += 1
                        if (pinch_release_frames >= MIN_PINCH_RELEASE and
                                current_time >= pinch_cooldown_until and
                                action_lock is None):
                            pinch_released = True
                    elif stable_gesture == "PINCH":
                        pinch_release_frames = 0
                    else:
                        pinch_confirm_frames = 0

                    # Track physical release of right click
                    # (Only re-arm when hand is not in 2-finger, 3-finger, 4-finger, or open-palm gestures)
                    if stable_gesture != "RIGHT CLICK":
                        right_click_confirm_frames = 0
                    if stable_gesture not in ("RIGHT CLICK", "DOUBLE CLICK", "FOUR FINGERS", "OPEN PALM"):
                        if current_time >= right_click_cooldown_until and action_lock is None:
                            right_click_released = True

                    # Track physical release of double click
                    if stable_gesture != "DOUBLE CLICK":
                        double_click_confirm_frames = 0
                    if stable_gesture not in ("DOUBLE CLICK", "FOUR FINGERS", "OPEN PALM"):
                        if current_time >= double_click_cooldown_until and action_lock is None:
                            double_click_released = True

                    # Track physical release of four fingers
                    # Requires confirmed neutral posture (POINT, OPEN PALM, FIST, or NO HAND)
                    if stable_gesture in ("POINT", "OPEN PALM", "FIST", "NO HAND"):
                        four_fingers_confirm_frames = 0
                        four_fingers_release_frames += 1
                        if (four_fingers_release_frames >= MIN_FOUR_FINGERS_RELEASE and
                                current_time >= four_fingers_cooldown_until and
                                action_lock is None):
                            four_fingers_released = True
                    elif stable_gesture == "FOUR FINGERS":
                        four_fingers_release_frames = 0
                    else:
                        four_fingers_confirm_frames = 0

                    # Reset scrolling tracking when not in POINT
                    if stable_gesture != "POINT":
                        scroll_start_y = None
                        scroll_cooldown_until = 0.0
                        last_scroll_direction = None
                        last_scroll_dir = None
                        scroll_deadzone_frames = 0

                    # 1. POINT -> Move Mouse Cursor & Touchless Vertical Scrolling
                    if stable_gesture == "POINT" and index_tip_pos is not None:
                        tip_x, tip_y = index_tip_pos
                        target_x = np.interp(tip_x, [margin_x, w - margin_x], [0, screen_w])
                        target_y = np.interp(tip_y, [margin_y, h - margin_y], [0, screen_h])
                        target_x = np.clip(target_x, 2, screen_w - 3)
                        target_y = np.clip(target_y, 2, screen_h - 3)

                        if prev_mouse_x is None or prev_mouse_y is None:
                            curr_mouse_x = target_x
                            curr_mouse_y = target_y
                        else:
                            curr_mouse_x = prev_mouse_x + (target_x - prev_mouse_x) / smooth_factor
                            curr_mouse_y = prev_mouse_y + (target_y - prev_mouse_y) / smooth_factor

                        prev_mouse_x, prev_mouse_y = curr_mouse_x, curr_mouse_y
                        pyautogui.moveTo(int(curr_mouse_x), int(curr_mouse_y))
                        control_status = "MOVING CURSOR"
                        cv2.circle(frame, (tip_x, tip_y), 10, (0, 255, 255), cv2.FILLED)

                        # Safety check: if cursor is over the AirTouch window, DO NOT scroll
                        if is_cursor_over_app_window(curr_mouse_x, curr_mouse_y, WINDOW_TITLE):
                            scroll_start_y = None
                            scroll_cooldown_until = 0.0
                            control_status = "SCROLL BLOCKED (OVER APP)"
                        else:
                            current_y = tip_y
                            if scroll_start_y is None:
                                scroll_start_y = current_y

                            now = time.monotonic()
                            delta_y = current_y - scroll_start_y

                            # 1. Recovery stroke absorption:
                            # When relaxing/returning after a scroll pulse, hand moves opposite to last scroll direction.
                            # Absorb opposite movement within the reversal window so recovery momentum does not trigger an accidental reversal.
                            if last_scroll_dir == "UP" and delta_y > 0:
                                if (current_time - last_scroll_dir_time < DIRECTION_REVERSAL_DELAY):
                                    scroll_start_y = current_y
                                    delta_y = 0
                            elif last_scroll_dir == "DOWN" and delta_y < 0:
                                if (current_time - last_scroll_dir_time < DIRECTION_REVERSAL_DELAY):
                                    scroll_start_y = current_y
                                    delta_y = 0

                            # 2. Deadzone stabilization:
                            # Absorb micro-tremors without resetting baseline every single frame.
                            # Baseline updates only after the hand remains stationary in deadzone for several frames.
                            if abs(delta_y) <= SCROLL_DEADZONE:
                                scroll_deadzone_frames += 1
                                if scroll_deadzone_frames >= 6:
                                    scroll_start_y = current_y
                                    if (current_time - last_scroll_dir_time >= DIRECTION_REVERSAL_DELAY):
                                        last_scroll_dir = None
                            else:
                                scroll_deadzone_frames = 0

                            # 3. Action execution:
                            if now >= scroll_cooldown_until:
                                if delta_y <= -SCROLL_TRIGGER_DISTANCE:
                                    # Intentional Scroll UP
                                    if last_scroll_dir == "DOWN" and (current_time - last_scroll_dir_time < DIRECTION_REVERSAL_DELAY):
                                        scroll_start_y = current_y
                                        scroll_cooldown_until = now + 0.15
                                    else:
                                        try:
                                            pyautogui.scroll(SCROLL_CLICKS)
                                            print("Action: Scroll UP")
                                            hud_action = "SCROLL UP"
                                            hud_action_expiry = current_time + ACTION_FEEDBACK_DURATION
                                            last_scroll_direction = "SCROLL UP"
                                            last_scroll_time = current_time
                                            last_scroll_dir = "UP"
                                            last_scroll_dir_time = current_time
                                        except Exception as e:
                                            print(f"Scroll UP failed: {e}")

                                        scroll_start_y = current_y
                                        scroll_cooldown_until = now + SCROLL_COOLDOWN

                                elif delta_y >= SCROLL_TRIGGER_DISTANCE:
                                    # Intentional Scroll DOWN
                                    if last_scroll_dir == "UP" and (current_time - last_scroll_dir_time < DIRECTION_REVERSAL_DELAY):
                                        scroll_start_y = current_y
                                        scroll_cooldown_until = now + 0.15
                                    else:
                                        try:
                                            pyautogui.scroll(-SCROLL_CLICKS)
                                            print("Action: Scroll DOWN")
                                            hud_action = "SCROLL DOWN"
                                            hud_action_expiry = current_time + ACTION_FEEDBACK_DURATION
                                            last_scroll_direction = "SCROLL DOWN"
                                            last_scroll_time = current_time
                                            last_scroll_dir = "DOWN"
                                            last_scroll_dir_time = current_time
                                        except Exception as e:
                                            print(f"Scroll DOWN failed: {e}")

                                        scroll_start_y = current_y
                                        scroll_cooldown_until = now + SCROLL_COOLDOWN

                            # Retain scroll HUD status briefly for readability
                            if (current_time - last_scroll_time < 0.3) and last_scroll_direction is not None:
                                control_status = last_scroll_direction

                    # 2. PINCH -> Exactly one left click, then transition to DRAG_ARMED
                    elif stable_gesture == "PINCH":
                        scroll_start_y = None
                        scroll_cooldown_until = 0.0
                        pinch_confirm_frames += 1
                        # Strictly blocked if an action lock is active (e.g. DOUBLE_CLICK or RIGHT_CLICK)
                        if action_lock is None and pinch_confirm_frames >= MIN_PINCH_CONFIRM and pinch_released and (current_time >= pinch_cooldown_until):
                            try:
                                current_pos = pyautogui.position()
                                cur_x, cur_y = current_pos.x, current_pos.y

                                if is_cursor_over_app_window(cur_x, cur_y, WINDOW_TITLE):
                                    control_status = "CLICK BLOCKED (OVER APP)"
                                    print("Safety: Click blocked because cursor is over AirTouch window.")
                                else:
                                    pyautogui.click()
                                    print(f"Action: Left Click triggered at ({cur_x}, {cur_y}).")
                                    hud_action = "LEFT CLICK"
                                    hud_action_expiry = current_time + ACTION_FEEDBACK_DURATION
                                    control_status = "CLICK (PINCHED)"

                                action_lock = "PINCH"
                                action_lock_time = current_time
                                pinch_released = False
                                pinch_release_frames = 0
                                pinch_cooldown_until = current_time + PINCH_COOLDOWN
                                # Transition immediately to DRAG_ARMED
                                app_state = "DRAG_ARMED"
                                drag_armed_time = current_time
                                fist_confirm_frames = 0
                            except Exception as click_err:
                                print(f"Exception during left click: {click_err}")
                                control_status = "CLICK ERROR"
                        else:
                            control_status = "CLICK (PINCHED)"

                    # 3. RIGHT CLICK -> Two Fingers (Index + Middle extended)
                    elif stable_gesture == "RIGHT CLICK":
                        scroll_start_y = None
                        scroll_cooldown_until = 0.0
                        double_click_confirm_frames = 0
                        right_click_confirm_frames += 1
                        # Strictly blocked if action_lock is active (e.g. from Double Click or Pinch)
                        if (action_lock is None and
                                right_click_confirm_frames >= MIN_RIGHT_CLICK_CONFIRM and
                                right_click_released and
                                (current_time >= right_click_cooldown_until)):
                            try:
                                current_pos = pyautogui.position()
                                cur_x, cur_y = current_pos.x, current_pos.y

                                if is_cursor_over_app_window(cur_x, cur_y, WINDOW_TITLE):
                                    control_status = "CLICK BLOCKED (OVER APP)"
                                    print("Safety: Right Click blocked because cursor is over AirTouch window.")
                                else:
                                    pyautogui.rightClick()
                                    print(f"Action: Right Click triggered at ({cur_x}, {cur_y})")
                                    hud_action = "RIGHT CLICK"
                                    hud_action_expiry = current_time + ACTION_FEEDBACK_DURATION
                                    control_status = "RIGHT CLICK"

                                action_lock = "RIGHT_CLICK"
                                action_lock_time = current_time
                                right_click_released = False
                                right_click_cooldown_until = current_time + RIGHT_CLICK_COOLDOWN
                            except Exception as right_err:
                                print(f"Exception during right click: {right_err}")
                                control_status = "RIGHT CLICK ERROR"
                        else:
                            control_status = "RIGHT CLICK"

                    # 4. DOUBLE CLICK -> Three Fingers (Index + Middle + Ring extended)
                    elif stable_gesture == "DOUBLE CLICK":
                        scroll_start_y = None
                        scroll_cooldown_until = 0.0
                        right_click_confirm_frames = 0
                        pinch_confirm_frames = 0

                        # Transition protection: if hand is moving into FOUR FINGERS, suppress DOUBLE CLICK
                        if raw_gesture == "FOUR FINGERS" or "FOUR FINGERS" in gesture_history:
                            double_click_confirm_frames = 0
                        else:
                            double_click_confirm_frames += 1

                        # Strictly blocked if another action lock is active
                        if (action_lock is None and
                                double_click_confirm_frames >= MIN_DOUBLE_CLICK_CONFIRM and
                                double_click_released and
                                (current_time >= double_click_cooldown_until)):
                            try:
                                current_pos = pyautogui.position()
                                cur_x, cur_y = current_pos.x, current_pos.y

                                if is_cursor_over_app_window(cur_x, cur_y, WINDOW_TITLE):
                                    control_status = "CLICK BLOCKED (OVER APP)"
                                    print("Safety: Double Click blocked because cursor is over AirTouch window.")
                                else:
                                    pyautogui.doubleClick()
                                    print(f"Action: Double Click triggered at ({cur_x}, {cur_y})")
                                    hud_action = "DOUBLE CLICK"
                                    hud_action_expiry = current_time + ACTION_FEEDBACK_DURATION
                                    control_status = "DOUBLE CLICK"

                                action_lock = "DOUBLE_CLICK"
                                action_lock_time = current_time
                                double_click_released = False
                                double_click_cooldown_until = current_time + DOUBLE_CLICK_COOLDOWN
                                # Also explicitly suppress right click during and after double click
                                right_click_released = False
                                right_click_cooldown_until = current_time + 0.6
                            except Exception as dbl_err:
                                print(f"Exception during double click: {dbl_err}")
                                control_status = "DOUBLE CLICK ERROR"
                        else:
                            control_status = "DOUBLE CLICK"

                    # 5. FOUR FINGERS -> Browser Back Navigation (Alt + Left)
                    elif stable_gesture == "FOUR FINGERS":
                        scroll_start_y = None
                        scroll_cooldown_until = 0.0
                        double_click_confirm_frames = 0
                        right_click_confirm_frames = 0
                        pinch_confirm_frames = 0

                        four_fingers_confirm_frames += 1
                        if (action_lock is None and
                                four_fingers_confirm_frames >= MIN_FOUR_FINGERS_CONFIRM and
                                four_fingers_released and
                                (current_time >= four_fingers_cooldown_until)):
                            try:
                                current_pos = pyautogui.position()
                                cur_x, cur_y = current_pos.x, current_pos.y

                                if is_cursor_over_app_window(cur_x, cur_y, WINDOW_TITLE):
                                    control_status = "NAV BLOCKED (OVER APP)"
                                    print("Safety: Navigation blocked because cursor is over AirTouch window.")
                                else:
                                    pyautogui.hotkey('alt', 'left')
                                    print("Action: Browser Back (Alt+Left)")
                                    hud_action = "BROWSER BACK"
                                    hud_action_expiry = current_time + ACTION_FEEDBACK_DURATION
                                    control_status = "BROWSER BACK"

                                action_lock = "FOUR_FINGERS"
                                action_lock_time = current_time
                                four_fingers_released = False
                                four_fingers_release_frames = 0
                                four_fingers_cooldown_until = current_time + FOUR_FINGERS_COOLDOWN
                                # Also explicitly suppress right click and double click during/after navigation
                                right_click_released = False
                                right_click_cooldown_until = current_time + 0.8
                                double_click_released = False
                                double_click_cooldown_until = current_time + 0.8
                            except Exception as nav_err:
                                print(f"Exception during browser navigation: {nav_err}")
                                control_status = "NAV ERROR"
                        else:
                            control_status = "BROWSER BACK"

                    # 6. OPEN PALM -> Pause Control
                    elif stable_gesture == "OPEN PALM":
                        prev_mouse_x, prev_mouse_y = None, None
                        control_status = "CONTROL PAUSED"

                    # 6. FIST in NORMAL -> Strictly IDLE
                    elif stable_gesture == "FIST":
                        prev_mouse_x, prev_mouse_y = None, None
                        control_status = "IDLE (FIST)"

                # ==============================================================
                # STATE 2: DRAG_ARMED (Waiting for FIST to start drag)
                # ==============================================================
                elif app_state == "DRAG_ARMED":
                    scroll_start_y = None
                    scroll_cooldown_until = 0.0
                    # Maintain non-PINCH release tracking if gesture left PINCH
                    if stable_gesture != "PINCH":
                        pinch_confirm_frames = 0

                    # Check arming timeout: if 2.0s elapsed without FIST, return to NORMAL
                    if current_time - drag_armed_time > DRAG_ARM_TIMEOUT:
                        app_state = "NORMAL"
                        fist_confirm_frames = 0
                        pinch_release_frames = 0
                        control_status = "IDLE"
                    elif stable_gesture == "FIST":
                        fist_confirm_frames += 1
                        if fist_confirm_frames >= MIN_FIST_CONFIRM:
                            try:
                                pyautogui.mouseDown(button='left')
                                print("Action: Drag started.")
                                hud_action = "DRAG START"
                                hud_action_expiry = current_time + ACTION_FEEDBACK_DURATION
                                app_state = "DRAGGING"
                                fist_release_frames = 0
                                fist_confirm_frames = 0
                                cur_pos = pyautogui.position()
                                prev_mouse_x, prev_mouse_y = cur_pos.x, cur_pos.y
                                control_status = "DRAGGING"
                            except Exception as drag_err:
                                print(f"Exception starting drag: {drag_err}")
                                app_state = "NORMAL"
                                control_status = "IDLE (FIST)"
                        else:
                            control_status = "IDLE (FIST)"
                    elif stable_gesture in ("OPEN PALM", "POINT", "RIGHT CLICK", "DOUBLE CLICK", "FOUR FINGERS"):
                        # User opened hand or moved away: cancel arming cleanly
                        app_state = "NORMAL"
                        fist_confirm_frames = 0
                        pinch_release_frames = 0
                        four_fingers_release_frames = 0
                        action_lock = None
                        if stable_gesture == "OPEN PALM":
                            control_status = "CONTROL PAUSED"
                        elif stable_gesture == "POINT":
                            control_status = "MOVING CURSOR"
                        elif stable_gesture == "RIGHT CLICK":
                            control_status = "RIGHT CLICK"
                        elif stable_gesture == "DOUBLE CLICK":
                            control_status = "DOUBLE CLICK"
                        else:
                            control_status = "BROWSER BACK"
                    else:
                        control_status = "CLICK (PINCHED)"

                # ==============================================================
                # STATE 3: DRAGGING (Active drag with left button down)
                # ==============================================================
                elif app_state == "DRAGGING":
                    scroll_start_y = None
                    scroll_cooldown_until = 0.0
                    # While dragging, cursor tracks hand position
                    if hand_drag_pos is not None:
                        drag_x, drag_y = hand_drag_pos
                        target_x = np.interp(drag_x, [margin_x, w - margin_x], [0, screen_w])
                        target_y = np.interp(drag_y, [margin_y, h - margin_y], [0, screen_h])
                        target_x = np.clip(target_x, 2, screen_w - 3)
                        target_y = np.clip(target_y, 2, screen_h - 3)

                        if prev_mouse_x is None or prev_mouse_y is None:
                            curr_mouse_x = target_x
                            curr_mouse_y = target_y
                        else:
                            curr_mouse_x = prev_mouse_x + (target_x - prev_mouse_x) / smooth_factor
                            curr_mouse_y = prev_mouse_y + (target_y - prev_mouse_y) / smooth_factor

                        prev_mouse_x, prev_mouse_y = curr_mouse_x, curr_mouse_y
                        pyautogui.moveTo(int(curr_mouse_x), int(curr_mouse_y))
                        cv2.circle(frame, (drag_x, drag_y), 12, (0, 165, 255), cv2.FILLED)

                    control_status = "DRAGGING"

                    # Release logic: require MIN_FIST_RELEASE frames away from FIST to confirm drop
                    if stable_gesture == "FIST":
                        fist_release_frames = 0
                    else:
                        fist_release_frames += 1
                        if fist_release_frames >= MIN_FIST_RELEASE:
                            # State transition: DRAGGING -> NORMAL
                            try:
                                pyautogui.mouseUp(button='left')
                                print("Action: Drag ended.")
                                hud_action = "DRAG END"
                                hud_action_expiry = current_time + ACTION_FEEDBACK_DURATION
                            except Exception as up_err:
                                print(f"Exception ending drag: {up_err}")

                            app_state = "NORMAL"
                            action_lock = None
                            fist_release_frames = 0
                            pinch_released = (stable_gesture != "PINCH")

            except Exception as mouse_err:
                # Any mouse control exception is caught and logged, never terminating the camera loop
                print(f"Non-fatal mouse control exception: {mouse_err}")

            # --- Professional On-screen Display (HUD) ---
            # 1. Determine Tracking & System Status
            tracking_status = "ACTIVE" if results.multi_hand_landmarks else "NO HAND"

            if tracking_status == "NO HAND":
                current_status = "IDLE"
            elif "BLOCKED" in control_status:
                current_status = "BLOCKED"
            elif app_state == "DRAGGING":
                current_status = "DRAGGING"
            elif app_state == "DRAG_ARMED":
                current_status = "ARMED"
            elif stable_gesture == "OPEN PALM":
                current_status = "PAUSED"
            elif stable_gesture == "FIST":
                current_status = "IDLE"
            elif stable_gesture in ("POINT", "PINCH", "RIGHT CLICK", "DOUBLE CLICK"):
                current_status = "ACTIVE"
            else:
                current_status = "IDLE"

            # 2. Determine Action to Display (Feedback persists briefly)
            if current_time <= hud_action_expiry:
                display_action = hud_action
            else:
                display_action = "NONE"

            # 3. Primary Status Card (Top-Left)
            draw_glass_panel(frame, (12, 12), (255, 142), bg_color=(20, 20, 20), border_color=(65, 65, 65), alpha=0.75)

            # Header: AIRTOUCH
            cv2.putText(frame, "AIRTOUCH", (22, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 220, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (20, 43), (247, 43), (60, 60, 60), 1, cv2.LINE_AA)

            # Tracking Status Line
            tracking_col = (60, 230, 80) if tracking_status == "ACTIVE" else (100, 100, 220)
            cv2.putText(frame, "Tracking :", (22, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, tracking_status, (100, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.46, tracking_col, 2 if tracking_status == "ACTIVE" else 1, cv2.LINE_AA)

            # Gesture Line
            gesture_col = GESTURE_COLORS.get(stable_gesture, (200, 200, 200))
            cv2.putText(frame, "Gesture  :", (22, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, stable_gesture, (100, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.46, gesture_col, 2, cv2.LINE_AA)

            # Status Line
            status_col = STATUS_COLORS.get(current_status, (200, 200, 200))
            cv2.putText(frame, "Status   :", (22, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, current_status, (100, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_col, 2, cv2.LINE_AA)

            # Action Line
            action_col = ACTION_COLORS.get(display_action, (150, 150, 150))
            action_thickness = 2 if display_action != "NONE" else 1
            cv2.putText(frame, "Action   :", (22, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, display_action, (100, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.46, action_col, action_thickness, cv2.LINE_AA)

            # 4. Compact Gesture Legend Card (Top-Right)
            draw_glass_panel(frame, (428, 12), (628, 160), bg_color=(20, 20, 20), border_color=(65, 65, 65), alpha=0.75)
            cv2.putText(frame, "GESTURE LEGEND", (438, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.line(frame, (436, 36), (620, 36), (60, 60, 60), 1, cv2.LINE_AA)

            legend_entries = [
                ("POINT", "-> Move / Scroll", (0, 230, 255)),
                ("PINCH", "-> Left Click", (255, 105, 180)),
                ("2 FINGERS", "-> Right Click", (255, 80, 255)),
                ("3 FINGERS", "-> Double Click", (255, 230, 0)),
                ("PINCH+FIST", "-> Drag", (0, 165, 255)),
                ("OPEN PALM", "-> Pause", (60, 230, 80)),
                ("FIST", "-> Idle", (50, 130, 255))
            ]

            leg_y = 52
            for g_name, g_desc, g_col in legend_entries:
                cv2.putText(frame, g_name, (438, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, g_col, 1, cv2.LINE_AA)
                cv2.putText(frame, g_desc, (522, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1, cv2.LINE_AA)
                leg_y += 15

            # 5. Subtle Footer Bar
            cv2.putText(frame, "AirTouch Touchless Control  |  Press 'q' to quit", (15, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

            # Show live webcam window
            cv2.imshow(WINDOW_TITLE, frame)

            # Listen for 'q' key to quit - only exit on explicit 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User pressed 'q'. Exiting application...")
                break

    except Exception as loop_err:
        print(f"Unexpected loop exception: {loop_err}")
        import traceback
        traceback.print_exc()

    finally:
        # Safety: Release mouse drag if still active
        if app_state == "DRAGGING":
            try:
                pyautogui.mouseUp(button='left')
                print("Action: Drag ended (cleanup).")
            except Exception:
                pass
            app_state = "NORMAL"

        # Clean up resources only after main loop exits
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
