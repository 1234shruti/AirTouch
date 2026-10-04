# AirTouch — Touchless Mouse Control

AirTouch is a computer-vision-based touchless mouse control system that enables users to control cursor movement and execute mouse actions using hand gestures captured through a standard webcam. By processing video frames in real time and mapping landmark positions to system mouse events, AirTouch provides an intuitive and contact-free human-computer interaction experience. The system incorporates state-machine debouncing and transition locking to deliver smooth, reliable, and conflict-free control.

## Features

- **Real-Time Hand Tracking**: Fast and accurate landmark detection using MediaPipe Hands running at interactive webcam framerates.
- **Cursor Movement (POINT)**: Smooth, jitter-free mouse cursor tracking mapped from the index fingertip with customizable smoothing factors.
- **Left Click (PINCH)**: Natural thumb-and-index pinch activation with frame confirmation and latching to prevent repeated clicks while holding.
- **Right Click (TWO FINGERS)**: Dedicated peace/two-finger gesture evaluated with multi-frame confirmation to prevent accidental triggers.
- **Double Click (THREE FINGERS)**: Three-finger extension gesture triggering a double click without triggering intermediate right or left clicks.
- **Drag & Drop (PINCH → FIST)**: Seamless state-driven drag and drop initiated by pinching and then closing into a fist, followed by clean release upon opening the hand.
- **Touchless Vertical Scrolling**: Vertical index finger movements with deadzone dampening and recovery stroke absorption for smooth upward and downward scrolling.
- **Pause Control (OPEN PALM)**: Instant control pause when showing an open palm, allowing users to move hands freely without displacing the cursor.
- **Idle State (FIST)**: Neutral closed fist state providing a safe idle position without triggering clicks or drags.
- **Gesture Debouncing & Transition Protection**: Priority-based action locking and rolling-buffer consensus that eliminate transition race conditions between 1-, 2-, and 3-finger hand shapes.
- **Camera-Window Safety Protection**: Automatic boundary detection that prevents clicks or scroll events from targeting the AirTouch camera application window.
- **Professional Real-Time HUD**: Glassmorphic on-screen display with live tracking status, active gesture indicators, dynamic state labels, action feedback notifications, and a compact gesture reference guide.

## Gesture Controls

| Gesture | Action | Description |
|---|---|---|
| **POINT** | Move Cursor / Scroll | Index finger extended; tracks cursor movement and vertical scrolling. |
| **PINCH** | Left Click | Thumb and index tips brought together; triggers exactly one left click. |
| **TWO FINGERS** | Right Click | Index and middle fingers extended (Peace Sign); triggers a single right click. |
| **THREE FINGERS** | Double Click | Index, middle, and ring fingers extended; triggers a single double click. |
| **PINCH → FIST** | Drag & Drop | Pinch to arm drag, close into fist to hold left mouse button, open hand to drop. |
| **OPEN PALM** | Pause | All fingers extended; pauses mouse tracking safely. |
| **FIST** | Idle | All fingers curled inward; idle state with no mouse actions. |

## Technology Stack

- **Python**: Core programming language powering the application logic, state machine, and test suites.
- **OpenCV (`opencv-python`)**: Video capture from webcam (DirectShow backend on Windows), image pre-processing, coordinate transformations, and HUD rendering.
- **MediaPipe (`mediapipe`)**: Machine-learning pipeline for 21 3D hand landmark estimation from live camera frames.
- **NumPy (`numpy`)**: Vectorized Euclidean geometry calculations, distance normalization, and coordinate interpolation.
- **PyAutoGUI (`pyautogui`)**: Cross-platform system automation library executing cursor movements, mouse clicks, mouse-wheel scrolling, and drag events.

## How It Works

The AirTouch pipeline processes frames continuously in a closed feedback loop:

```text
Webcam
  → OpenCV (Frame Capture & BGR-to-RGB Conversion)
  → MediaPipe Hand Tracking (21 3D Landmark Estimation)
  → Finger / Landmark Analysis (Joint Angles, Extension, & Normalized Distance Checks)
  → Gesture Classification (POINT, PINCH, TWO FINGERS, THREE FINGERS, OPEN PALM, FIST)
  → Gesture Debouncing / State Machine (Multi-Frame Consensus, Action Locks, Cooldowns)
  → PyAutoGUI (Operating System Mouse Movement & Action Execution)
  → Mouse Action (Screen Cursor Updates, Click, Drag, or Scroll)
```

## Gesture Reliability

Reliability is critical for touchless interaction because hands naturally transition through intermediate shapes (for example, raising three fingers passes briefly through a two-finger position). AirTouch solves this using several robust safeguards:

- **Multi-Frame Confirmation**: Requires consecutive frames of a detected gesture before confirming it (e.g., 4 frames for Right Click, 3 frames for Double Click, 2 frames for Pinch), filtering out fleeting sensor noise.
- **Cooldown Timers**: Enforces minimum intervals between click activations (`0.5s` for pinch, `0.6s` for right/double clicks) so held gestures never machine-gun repeat clicks.
- **Action Locks**: When an action like Double Click fires, the system engages an action lock that suppresses subordinate gestures (such as Right Click or Pinch) until the hand is released.
- **Neutral-Release Detection**: Unlocking requires the user to return to a neutral resting state (POINT, OPEN PALM, or FIST) for at least 3 consecutive frames after an action.
- **Transition Protection**: Three-finger unfolding and folding cannot accidentally trigger two-finger right clicks during intermediate movement phases.
- **Scroll Deadzone**: A `±20px` baseline deadzone absorbs physiological hand tremor without drifting the scroll origin.
- **Direction Reversal Protection**: A `0.45s` directional delay absorbs hand recovery strokes, preventing unwanted bounce-back scrolling in the opposite direction.

## Safety

AirTouch incorporates active window boundary protection to prevent recursion or UI lockup:
- The system checks current cursor coordinates against the OpenCV window rectangle using Windows API calls.
- If the mouse cursor hovers over the AirTouch camera window or title bar, click and scroll commands are blocked.
- Terminal output logs `Safety: Click blocked because cursor is over AirTouch window.`, ensuring the user never accidentally minimizes, resizes, or closes the camera feed during control.

## Installation

Clone the repository and set up a virtual environment using Windows PowerShell:

```powershell
git clone <repository-url>
cd AirTouch
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

To start the application:

```powershell
.\venv\Scripts\python.exe main.py
```

- Position your hand comfortably within the bounding box guide shown in the webcam feed.
- Press **`q`** on the keyboard while focused on the camera window to safely release any active drag states, free the camera hardware, and exit the application.

## Testing

AirTouch includes automated unit and state simulation tests verifying classification exclusivity and multi-frame transition safety:

```powershell
# Verify syntax and compilation
.\venv\Scripts\python.exe -m py_compile main.py

# Run the automated gesture transition and conflict test suite
.\venv\Scripts\python.exe test_gestures.py
```

The test suite validates 8 comprehensive verification scenarios:
1. Gesture classification & strict mutual exclusivity
2. Three Fingers $\rightarrow$ Double Click only (0 Right Clicks on raise and lower)
3. Three Fingers $\rightarrow$ 0 Left Clicks during pinch-like transition shapes
4. Three Fingers $\rightarrow$ FIST does not trigger Drag
5. Two Fingers $\rightarrow$ Right Click only (0 Double Clicks, 0 Left Clicks)
6. PINCH $\rightarrow$ Exactly one Left Click with hold suppression
7. PINCH $\rightarrow$ FIST $\rightarrow$ Drag starts and ends cleanly without trailing clicks
8. Holding Double Click / Right Click produces exactly 1 action

## Project Structure

```text
AirTouch/
├── main.py              # Main application entry point, gesture engine, HUD, and control loop
├── test_gestures.py     # Automated unit and state machine transition test suite (8 tests)
├── test_scroll.py       # Standalone Windows mouse wheel event diagnostic utility
├── requirements.txt     # Python project dependencies
└── README.md            # Project documentation and user guide
```

## Future Scope

The following ideas represent potential future enhancements:

- **Configurable Gestures**: Allowing users to map gestures to custom keyboard shortcuts or alternative mouse buttons.
- **Additional Hand Gestures**: Adding distinct gestures for browser navigation (forward/back) and audio volume control.
- **Customizable Sensitivity**: UI slider controls to adjust cursor smoothing, interaction margins, and scroll acceleration.
- **Multi-Hand Support**: Dual-hand interaction (e.g., left hand for mode switching, right hand for pointing).
- **Application-Specific Gesture Profiles**: Automatically switching sensitivity and bindings depending on the active foreground application (e.g., presentation slides vs. web browsing).

## Credits / Author

**Shruti Jumbad**
