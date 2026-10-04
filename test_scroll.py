import ctypes
import time
import sys
import msvcrt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

user32 = ctypes.windll.user32

MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120

print("Windows mouse wheel test", flush=True)
print("Move your mouse/click the browser window so it is the active window.", flush=True)
print("Press:", flush=True)
print("u = scroll UP", flush=True)
print("d = scroll DOWN", flush=True)
print("q = quit", flush=True)

VK_U = 0x55  # 'U' key
VK_D = 0x44  # 'D' key
VK_Q = 0x51  # 'Q' key

u_was_down = False
d_was_down = False

try:
    while True:
        # Check global key state so test works even when browser window is the active window
        u_down = bool(user32.GetAsyncKeyState(VK_U) & 0x8000)
        d_down = bool(user32.GetAsyncKeyState(VK_D) & 0x8000)
        q_down = bool(user32.GetAsyncKeyState(VK_Q) & 0x8000)

        # Also drain console input if typed directly into terminal
        if msvcrt.kbhit():
            try:
                ch = msvcrt.getch().lower()
                if ch == b'u':
                    u_down = True
                elif ch == b'd':
                    d_down = True
                elif ch == b'q':
                    q_down = True
            except Exception:
                pass

        if q_down:
            print("Exiting...", flush=True)
            sys.exit(0)

        if u_down and not u_was_down:
            user32.mouse_event(
                MOUSEEVENTF_WHEEL,
                0,
                0,
                +120,
                0
            )
            print("Sent physical wheel UP test", flush=True)

        if d_down and not d_was_down:
            user32.mouse_event(
                MOUSEEVENTF_WHEEL,
                0,
                0,
                -120,
                0
            )
            print("Sent physical wheel DOWN test", flush=True)

        u_was_down = u_down
        d_was_down = d_down
        time.sleep(0.05)

except KeyboardInterrupt:
    sys.exit(0)
