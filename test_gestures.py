"""
Unit and state simulation tests for AirTouch gesture control.
Tests:
1. Gesture detection classification and strict mutual exclusivity
2. Three Fingers -> Double Click only (with 2-finger and pinch-like transition frames)
3. Three Fingers -> Fist -> NO Drag
4. Two Fingers -> Right Click only
5. Pinch -> Exactly one Left Click (holding pinch does not repeat)
6. Pinch -> Fist -> Drag starts and ends cleanly
7. Holding any click gesture does not repeat
"""
from collections import Counter, deque
from main import detect_gesture

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def create_hand(index_open=False, middle_open=False, ring_open=False, pinky_open=False, thumb_open=False, pinch=False):
    # 21 landmarks
    lms = [MockLandmark(0.5, 0.8) for _ in range(21)]
    lms[0] = MockLandmark(0.5, 0.8)   # Wrist
    lms[9] = MockLandmark(0.5, 0.5)   # Middle MCP -> palm_size = 0.3
    lms[2] = MockLandmark(0.4, 0.7)   # Thumb MCP
    lms[17] = MockLandmark(0.65, 0.6) # Pinky base

    # Thumb
    if pinch:
        lms[4] = MockLandmark(0.45, 0.3)
        lms[8] = MockLandmark(0.45, 0.32) # normalized distance = 0.02 / 0.3 = 0.067 (< 0.35)
    elif thumb_open:
        lms[4] = MockLandmark(0.2, 0.5)
    else:
        lms[4] = MockLandmark(0.5, 0.65)

    # Index: tip 8, pip 6
    lms[6] = MockLandmark(0.45, 0.45)
    if not pinch:
        lms[8] = MockLandmark(0.45, 0.2 if index_open else 0.65)

    # Middle: tip 12, pip 10
    lms[10] = MockLandmark(0.5, 0.45)
    lms[12] = MockLandmark(0.5, 0.2 if middle_open else 0.65)

    # Ring: tip 16, pip 14
    lms[14] = MockLandmark(0.55, 0.45)
    lms[16] = MockLandmark(0.55, 0.2 if ring_open else 0.65)

    # Pinky: tip 20, pip 18
    lms[18] = MockLandmark(0.6, 0.45)
    lms[20] = MockLandmark(0.6, 0.2 if pinky_open else 0.65)

    return lms

class StateMachineSimulator:
    """Exact simulation of the state machine logic in main.py."""
    def __init__(self):
        self.gesture_history = deque(maxlen=7)
        self.app_state = "NORMAL"
        self.action_lock = None
        self.action_lock_time = 0.0
        self.neutral_confirm_frames = 0
        self.MIN_NEUTRAL_CONFIRM = 3

        self.pinch_released = True
        self.pinch_cooldown_until = 0.0
        self.PINCH_COOLDOWN = 0.5
        self.pinch_confirm_frames = 0
        self.MIN_PINCH_CONFIRM = 2

        self.drag_armed_time = 0.0
        self.DRAG_ARM_TIMEOUT = 2.0
        self.fist_confirm_frames = 0
        self.MIN_FIST_CONFIRM = 2
        self.fist_release_frames = 0
        self.MIN_FIST_RELEASE = 4

        self.right_click_released = True
        self.right_click_cooldown_until = 0.0
        self.RIGHT_CLICK_COOLDOWN = 0.6
        self.right_click_confirm_frames = 0
        self.MIN_RIGHT_CLICK_CONFIRM = 4

        self.double_click_released = True
        self.double_click_cooldown_until = 0.0
        self.DOUBLE_CLICK_COOLDOWN = 0.6
        self.double_click_confirm_frames = 0
        self.MIN_DOUBLE_CLICK_CONFIRM = 4

        self.four_fingers_released = True
        self.four_fingers_cooldown_until = 0.0
        self.FOUR_FINGERS_COOLDOWN = 0.8
        self.four_fingers_confirm_frames = 0
        self.MIN_FOUR_FINGERS_CONFIRM = 3
        self.four_fingers_release_frames = 0
        self.MIN_FOUR_FINGERS_RELEASE = 3

        # Counters for verified event triggers
        self.left_clicks = 0
        self.right_clicks = 0
        self.double_clicks = 0
        self.browser_backs = 0
        self.drag_starts = 0
        self.drag_ends = 0
        self.current_time = 1000.0

    def step(self, raw_gesture):
        self.current_time += 0.033  # ~30fps
        self.gesture_history.append(raw_gesture)
        stable_gesture = Counter(self.gesture_history).most_common(1)[0][0]

        # --- NORMAL State Machine ---
        if self.app_state == "NORMAL":
            # Neutral tracking
            if stable_gesture in ("POINT", "OPEN PALM", "FIST", "NO HAND"):
                self.neutral_confirm_frames += 1
                if self.neutral_confirm_frames >= self.MIN_NEUTRAL_CONFIRM:
                    if self.action_lock in ("DOUBLE_CLICK", "RIGHT_CLICK", "PINCH", "FOUR_FINGERS"):
                        if self.current_time >= self.action_lock_time + 0.35:
                            self.action_lock = None
            else:
                self.neutral_confirm_frames = 0

            # Release tracking
            if stable_gesture != "PINCH":
                self.pinch_confirm_frames = 0
                if self.current_time >= self.pinch_cooldown_until and self.action_lock is None:
                    self.pinch_released = True

            if stable_gesture != "RIGHT CLICK":
                self.right_click_confirm_frames = 0
            if stable_gesture not in ("RIGHT CLICK", "DOUBLE CLICK", "FOUR FINGERS", "OPEN PALM"):
                if self.current_time >= self.right_click_cooldown_until and self.action_lock is None:
                    self.right_click_released = True

            if stable_gesture != "DOUBLE CLICK":
                self.double_click_confirm_frames = 0
            if stable_gesture not in ("DOUBLE CLICK", "FOUR FINGERS", "OPEN PALM"):
                if self.current_time >= self.double_click_cooldown_until and self.action_lock is None:
                    self.double_click_released = True

            # Track physical release of four fingers
            if stable_gesture in ("POINT", "OPEN PALM", "FIST", "NO HAND"):
                self.four_fingers_confirm_frames = 0
                self.four_fingers_release_frames += 1
                if (self.four_fingers_release_frames >= self.MIN_FOUR_FINGERS_RELEASE and
                        self.current_time >= self.four_fingers_cooldown_until and
                        self.action_lock is None):
                    self.four_fingers_released = True
            elif stable_gesture == "FOUR FINGERS":
                self.four_fingers_release_frames = 0
            else:
                self.four_fingers_confirm_frames = 0

            # Actions
            if stable_gesture == "PINCH":
                self.pinch_confirm_frames += 1
                if (self.action_lock is None and
                        self.pinch_confirm_frames >= self.MIN_PINCH_CONFIRM and
                        self.pinch_released and
                        self.current_time >= self.pinch_cooldown_until):
                    self.left_clicks += 1
                    self.action_lock = "PINCH"
                    self.action_lock_time = self.current_time
                    self.pinch_released = False
                    self.pinch_cooldown_until = self.current_time + self.PINCH_COOLDOWN
                    self.app_state = "DRAG_ARMED"
                    self.drag_armed_time = self.current_time
                    self.fist_confirm_frames = 0

            elif stable_gesture == "RIGHT CLICK":
                self.double_click_confirm_frames = 0
                self.right_click_confirm_frames += 1
                if (self.action_lock is None and
                        self.right_click_confirm_frames >= self.MIN_RIGHT_CLICK_CONFIRM and
                        self.right_click_released and
                        self.current_time >= self.right_click_cooldown_until):
                    self.right_clicks += 1
                    self.action_lock = "RIGHT_CLICK"
                    self.action_lock_time = self.current_time
                    self.right_click_released = False
                    self.right_click_cooldown_until = self.current_time + self.RIGHT_CLICK_COOLDOWN

            elif stable_gesture == "DOUBLE CLICK":
                self.right_click_confirm_frames = 0
                self.pinch_confirm_frames = 0
                # Transition protection: suppress double click if moving into FOUR FINGERS
                if raw_gesture == "FOUR FINGERS" or "FOUR FINGERS" in self.gesture_history:
                    self.double_click_confirm_frames = 0
                else:
                    self.double_click_confirm_frames += 1

                if (self.action_lock is None and
                        self.double_click_confirm_frames >= self.MIN_DOUBLE_CLICK_CONFIRM and
                        self.double_click_released and
                        self.current_time >= self.double_click_cooldown_until):
                    self.double_clicks += 1
                    self.action_lock = "DOUBLE_CLICK"
                    self.action_lock_time = self.current_time
                    self.double_click_released = False
                    self.double_click_cooldown_until = self.current_time + self.DOUBLE_CLICK_COOLDOWN
                    self.right_click_released = False
                    self.right_click_cooldown_until = self.current_time + 0.6

            elif stable_gesture == "FOUR FINGERS":
                self.double_click_confirm_frames = 0
                self.right_click_confirm_frames = 0
                self.pinch_confirm_frames = 0
                self.four_fingers_confirm_frames += 1
                if (self.action_lock is None and
                        self.four_fingers_confirm_frames >= self.MIN_FOUR_FINGERS_CONFIRM and
                        self.four_fingers_released and
                        self.current_time >= self.four_fingers_cooldown_until):
                    self.browser_backs += 1
                    self.action_lock = "FOUR_FINGERS"
                    self.action_lock_time = self.current_time
                    self.four_fingers_released = False
                    self.four_fingers_release_frames = 0
                    self.four_fingers_cooldown_until = self.current_time + self.FOUR_FINGERS_COOLDOWN
                    self.right_click_released = False
                    self.right_click_cooldown_until = self.current_time + 0.8
                    self.double_click_released = False
                    self.double_click_cooldown_until = self.current_time + 0.8

        elif self.app_state == "DRAG_ARMED":
            if stable_gesture != "PINCH":
                self.pinch_confirm_frames = 0
                if self.current_time >= self.pinch_cooldown_until:
                    self.pinch_released = True

            if self.current_time - self.drag_armed_time > self.DRAG_ARM_TIMEOUT:
                self.app_state = "NORMAL"
                self.fist_confirm_frames = 0
                self.action_lock = None
            elif stable_gesture == "FIST":
                self.fist_confirm_frames += 1
                if self.fist_confirm_frames >= self.MIN_FIST_CONFIRM:
                    self.drag_starts += 1
                    self.app_state = "DRAGGING"
                    self.fist_release_frames = 0
                    self.fist_confirm_frames = 0
            elif stable_gesture in ("OPEN PALM", "POINT", "RIGHT CLICK", "DOUBLE CLICK", "FOUR FINGERS"):
                self.app_state = "NORMAL"
                self.fist_confirm_frames = 0
                self.action_lock = None

        elif self.app_state == "DRAGGING":
            if stable_gesture == "FIST":
                self.fist_release_frames = 0
            else:
                self.fist_release_frames += 1
                if self.fist_release_frames >= self.MIN_FIST_RELEASE:
                    self.drag_ends += 1
                    self.app_state = "NORMAL"
                    self.action_lock = None
                    self.fist_release_frames = 0
                    self.pinch_released = (stable_gesture != "PINCH")

def test_gesture_classification():
    """Verify raw landmark classification and exclusivity."""
    fist = create_hand(index_open=False, middle_open=False, ring_open=False, pinky_open=False)
    assert detect_gesture(fist) == "FIST"

    point = create_hand(index_open=True, middle_open=False, ring_open=False, pinky_open=False)
    assert detect_gesture(point) == "POINT"

    two_fingers = create_hand(index_open=True, middle_open=True, ring_open=False, pinky_open=False)
    assert detect_gesture(two_fingers) == "RIGHT CLICK"
    assert detect_gesture(two_fingers) != "DOUBLE CLICK"
    assert detect_gesture(two_fingers) != "PINCH"

    three_fingers = create_hand(index_open=True, middle_open=True, ring_open=True, pinky_open=False)
    assert detect_gesture(three_fingers) == "DOUBLE CLICK"
    assert detect_gesture(three_fingers) != "RIGHT CLICK"
    assert detect_gesture(three_fingers) != "PINCH"

    palm = create_hand(index_open=True, middle_open=True, ring_open=True, pinky_open=True, thumb_open=True)
    assert detect_gesture(palm) == "OPEN PALM"

    four_fingers = create_hand(index_open=True, middle_open=True, ring_open=True, pinky_open=True, thumb_open=False)
    assert detect_gesture(four_fingers) == "FOUR FINGERS"
    assert detect_gesture(four_fingers) != "OPEN PALM"
    assert detect_gesture(four_fingers) != "DOUBLE CLICK"

    pinch = create_hand(index_open=True, pinch=True)
    assert detect_gesture(pinch) == "PINCH"
    assert detect_gesture(pinch) != "RIGHT CLICK"
    assert detect_gesture(pinch) != "DOUBLE CLICK"
    assert detect_gesture(pinch) != "FOUR FINGERS"
    print("PASS 1: Gesture classification & mutual exclusivity verified.")

def test_three_fingers_with_two_finger_transition():
    """Test 3 fingers: passing through 2 fingers for 2 frames during raise and lower."""
    sim = StateMachineSimulator()
    # Baseline neutral (POINT)
    for _ in range(7):
        sim.step("POINT")

    # Raising fingers: passes through RIGHT CLICK for 2 frames
    sim.step("RIGHT CLICK")
    sim.step("RIGHT CLICK")

    # Stably at DOUBLE CLICK for 8 frames
    for _ in range(8):
        sim.step("DOUBLE CLICK")

    # Lowering fingers: passes through RIGHT CLICK for 2 frames
    sim.step("RIGHT CLICK")
    sim.step("RIGHT CLICK")

    # Return to POINT neutral
    for _ in range(7):
        sim.step("POINT")

    assert sim.double_clicks == 1, f"Expected 1 Double Click, got {sim.double_clicks}"
    assert sim.right_clicks == 0, f"Expected 0 Right Clicks during 3-finger gesture, got {sim.right_clicks}"
    assert sim.left_clicks == 0, f"Expected 0 Left Clicks, got {sim.left_clicks}"
    print("PASS 2: Three Fingers -> Double Click ONLY (0 Right Clicks on raise/lower).")

def test_three_fingers_with_pinch_transition():
    """Test 3 fingers: temporary pinch-like frame during hand raise."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")

    # Transient pinch-like frame while hand is unfurling
    sim.step("PINCH")

    # 3 fingers extended
    for _ in range(10):
        sim.step("DOUBLE CLICK")

    for _ in range(7):
        sim.step("POINT")

    assert sim.double_clicks == 1, f"Expected 1 Double Click, got {sim.double_clicks}"
    assert sim.left_clicks == 0, f"Expected 0 Left Clicks during 3-finger unfurl, got {sim.left_clicks}"
    print("PASS 3: Three Fingers -> 0 Left Clicks during pinch-like transition.")

def test_three_fingers_then_fist_no_drag():
    """Test 3 fingers followed immediately by FIST: must NOT start drag."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")

    for _ in range(8):
        sim.step("DOUBLE CLICK")

    # User closes hand into FIST
    for _ in range(10):
        sim.step("FIST")

    assert sim.double_clicks == 1
    assert sim.drag_starts == 0, f"Expected 0 Drag Starts after Double Click, got {sim.drag_starts}"
    print("PASS 4: Three Fingers -> FIST does NOT trigger Drag.")

def test_two_fingers_right_click_only():
    """Test intentional Two Fingers (Peace sign) -> Right Click only."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")

    # User holds 2 fingers for 10 frames
    for _ in range(10):
        sim.step("RIGHT CLICK")

    for _ in range(7):
        sim.step("POINT")

    assert sim.right_clicks == 1, f"Expected 1 Right Click, got {sim.right_clicks}"
    assert sim.double_clicks == 0, f"Expected 0 Double Clicks, got {sim.double_clicks}"
    assert sim.left_clicks == 0, f"Expected 0 Left Clicks, got {sim.left_clicks}"
    print("PASS 5: Two Fingers -> Right Click ONLY (0 Double Clicks, 0 Left Clicks).")

def test_pinch_single_click_and_holding():
    """Test PINCH produces exactly 1 click and holding does not repeat."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")

    # Hold PINCH for 25 frames
    for _ in range(25):
        sim.step("PINCH")

    assert sim.left_clicks == 1, f"Expected exactly 1 Left Click while holding PINCH, got {sim.left_clicks}"

    # Return to POINT neutral
    for _ in range(10):
        sim.step("POINT")

    # Second physical PINCH
    for _ in range(6):
        sim.step("PINCH")

    assert sim.left_clicks == 2, f"Expected 2 total Left Clicks, got {sim.left_clicks}"
    print("PASS 6: PINCH -> Exactly one Left Click, holding does not repeat.")

def test_pinch_then_fist_drag():
    """Test PINCH -> FIST initiates drag and opening hand ends drag."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")

    # PINCH
    for _ in range(5):
        sim.step("PINCH")
    assert sim.left_clicks == 1
    assert sim.app_state == "DRAG_ARMED"

    # FIST
    for _ in range(5):
        sim.step("FIST")
    assert sim.drag_starts == 1, "Drag should start on confirmed FIST"
    assert sim.app_state == "DRAGGING"

    # OPEN PALM
    for _ in range(8):
        sim.step("OPEN PALM")
    assert sim.drag_ends == 1, "Drag should end on confirmed OPEN PALM"
    assert sim.app_state == "NORMAL"
    print("PASS 7: PINCH -> FIST -> Drag starts and ends cleanly.")

def test_holding_click_gestures_no_repeat():
    """Test holding Double Click or Right Click for 30 frames produces exactly 1 trigger."""
    # Test holding Double Click
    sim_dbl = StateMachineSimulator()
    for _ in range(7):
        sim_dbl.step("POINT")
    for _ in range(30):
        sim_dbl.step("DOUBLE CLICK")
    assert sim_dbl.double_clicks == 1, f"Expected 1 Double Click, got {sim_dbl.double_clicks}"

    # Test holding Right Click
    sim_rc = StateMachineSimulator()
    for _ in range(7):
        sim_rc.step("POINT")
    for _ in range(30):
        sim_rc.step("RIGHT CLICK")
    assert sim_rc.right_clicks == 1, f"Expected 1 Right Click, got {sim_rc.right_clicks}"
    print("PASS 8: Holding Double Click / Right Click produces exactly 1 action.")

def test_four_fingers_single_trigger():
    """Test 1: FOUR FINGERS produces exactly 1 Browser Back."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")
    for _ in range(10):
        sim.step("FOUR FINGERS")
    assert sim.browser_backs == 1, f"Expected 1 Browser Back, got {sim.browser_backs}"
    assert sim.double_clicks == 0
    assert sim.right_clicks == 0
    assert sim.left_clicks == 0
    print("PASS 9: FOUR FINGERS -> Exactly 1 Browser Back.")

def test_four_fingers_holding_no_repeat():
    """Test 2: Holding FOUR FINGERS for many frames produces exactly 1 Browser Back."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")
    for _ in range(35):
        sim.step("FOUR FINGERS")
    assert sim.browser_backs == 1, f"Expected 1 Browser Back while holding, got {sim.browser_backs}"
    print("PASS 10: Holding FOUR FINGERS produces exactly 1 Browser Back (no repeat).")

def test_four_fingers_release_and_repeat():
    """Test 3: FOUR FINGERS -> release to neutral -> FOUR FINGERS produces exactly 2 Browser Backs."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")
    # First Four Fingers
    for _ in range(10):
        sim.step("FOUR FINGERS")
    assert sim.browser_backs == 1

    # Release to neutral POINT for confirmed release frames + cooldown
    for _ in range(30):
        sim.step("POINT")

    # Second Four Fingers
    for _ in range(10):
        sim.step("FOUR FINGERS")
    assert sim.browser_backs == 2, f"Expected 2 Browser Backs, got {sim.browser_backs}"
    print("PASS 11: Release and repeat produces exactly 2 Browser Back actions.")

def test_three_to_four_transition():
    """Test 4: Three -> Four transition produces 1 Browser Back, 0 Double Click."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")

    # Passes through 3 fingers (DOUBLE CLICK) for 2 frames while raising 4th finger
    sim.step("DOUBLE CLICK")
    sim.step("DOUBLE CLICK")

    # Stably at FOUR FINGERS
    for _ in range(8):
        sim.step("FOUR FINGERS")

    for _ in range(10):
        sim.step("POINT")

    assert sim.browser_backs == 1, f"Expected 1 Browser Back, got {sim.browser_backs}"
    assert sim.double_clicks == 0, f"Expected 0 Double Clicks during Three->Four transition, got {sim.double_clicks}"
    assert sim.right_clicks == 0
    assert sim.left_clicks == 0
    print("PASS 12: Three -> Four transition produces 1 Browser Back, 0 Double Click.")

def test_four_to_three_transition():
    """Test 5: Four -> Three transition produces 0 Double Click during lowering."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")

    # Stably at FOUR FINGERS
    for _ in range(8):
        sim.step("FOUR FINGERS")
    assert sim.browser_backs == 1

    # Lowering hand: passes through DOUBLE CLICK for 2 frames
    sim.step("DOUBLE CLICK")
    sim.step("DOUBLE CLICK")

    # Return to POINT neutral
    for _ in range(10):
        sim.step("POINT")

    assert sim.browser_backs == 1
    assert sim.double_clicks == 0, f"Expected 0 Double Clicks during Four->Three lowering transition, got {sim.double_clicks}"
    print("PASS 13: Four -> Three lowering transition produces 0 Double Click.")

def test_genuine_three_fingers():
    """Test 6: Genuine Three Fingers held intentionally still produces 1 Double Click."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")
    for _ in range(10):
        sim.step("DOUBLE CLICK")
    for _ in range(10):
        sim.step("POINT")
    assert sim.double_clicks == 1, f"Expected 1 Double Click, got {sim.double_clicks}"
    assert sim.browser_backs == 0
    print("PASS 14: Genuine Three Fingers produces 1 Double Click.")

def test_four_fingers_no_conflicts():
    """Test 7 & 8: FOUR FINGERS produces 0 Right Click and 0 Left Click."""
    sim = StateMachineSimulator()
    for _ in range(7):
        sim.step("POINT")
    for _ in range(15):
        sim.step("FOUR FINGERS")
    assert sim.right_clicks == 0, f"Expected 0 Right Click, got {sim.right_clicks}"
    assert sim.left_clicks == 0, f"Expected 0 Left Click, got {sim.left_clicks}"
    print("PASS 15: FOUR FINGERS produces 0 Right Click and 0 Left Click.")

if __name__ == "__main__":
    test_gesture_classification()
    test_three_fingers_with_two_finger_transition()
    test_three_fingers_with_pinch_transition()
    test_three_fingers_then_fist_no_drag()
    test_two_fingers_right_click_only()
    test_pinch_single_click_and_holding()
    test_pinch_then_fist_drag()
    test_holding_click_gestures_no_repeat()
    test_four_fingers_single_trigger()
    test_four_fingers_holding_no_repeat()
    test_four_fingers_release_and_repeat()
    test_three_to_four_transition()
    test_four_to_three_transition()
    test_genuine_three_fingers()
    test_four_fingers_no_conflicts()
    print("\nALL TRANSITION, LOCKING & BROWSER BACK TESTS PASSED PERFECTLY!")
