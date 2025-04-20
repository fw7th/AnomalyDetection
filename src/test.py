import cv2
import time
import math
import numpy as np

def draw_pulsing_alert(frame, base_thickness=4, pulse_range=6, frequency=1):
    h, w = frame.shape[:2]
    t = time.time()
    
    # Generate sine wave
    sine_val = math.sin(2 * math.pi * frequency * t)  # -1 to 1
    pulse = abs(sine_val)  # 0 to 1

    # Pulsing thickness
    thickness = int(base_thickness + pulse_range * pulse)

    # Pulsing red color (fades from dark red to bright red)
    red_intensity = int(150 + 105 * pulse)  # 150 to 255
    color = (0, 0, red_intensity)

    # Draw border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

    # Draw alert text
    text = "ALERT!"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    text_thickness = 2 + int(pulse * 2)  # Thickness: 2 to 4
    text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 100  # Distance from top

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, text_thickness, cv2.LINE_AA)

    return frame

# Main loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    condition = True  # Your trigger logic

    if condition:
        frame = draw_pulsing_alert(frame)

    cv2.imshow("Synced Alert", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

