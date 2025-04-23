import cv2 as cv
import time, math, os, sys

def visual_alerts(frame, base_thickness=4, pulse_range=6, frequency=0.4):
    h, w = frame.shape[:2]
    t = time.time()
    
    # Generate sine wave
    pulse = (math.sin(2 * math.pi * frequency * t) + 1) / 2

    # Pulsing thickness
    thickness = int(base_thickness + pulse_range * pulse)

    # Pulsing red color (fades from dark red to bright red)
    red_intensity = int(150 + 105 * pulse)  # 150 to 255
    color = (0, 0, red_intensity)

    # Draw border
    cv.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

    # Draw alert text
    text = "ALERT!"
    font = cv.FONT_HERSHEY_DUPLEX
    font_scale = 2
    text_thickness = 2 + int(pulse * 2)  # Thickness: 2 to 4
    text_size = cv.getTextSize(text, font, font_scale, text_thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 100  # Distance from top

    cv.putText(frame, text, (text_x, text_y), font, font_scale, color, text_thickness, cv.LINE_AA)

    return frame

def sound_alerts():
    if sys.platform == "win32":
        # For windows systems
        import winsound
        winsound.Beep(1000, 500)
    else:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sound = os.path.join(BASE_DIR, "beep.wav")

        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(sound)  # Must be .wav
        # For Unix-like systems, the beep command
        os.system('echo -e "\a"')
        wave_obj.play()  # Non-blocking
