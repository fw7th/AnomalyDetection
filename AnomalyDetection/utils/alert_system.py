"""
alert_system.py

Implementation for the sound and visual alert systems.
"""

import cv2 as cv
import time, math, os, sys

def visual_alerts(frame, base_thickness=4, pulse_range=6, frequency=0.4):
    """
    Add visual alerts to the provided frame by displaying a pulsing red border
    and an alert message. The border thickness and color intensity pulsate based
    on the sine wave function to create a dynamic effect.
    
    Parameters:
    -----------
    frame : numpy.ndarray
        The image frame (as a NumPy array) to apply the visual alert to.
    
    base_thickness : int, optional, default: 4
        The base thickness of the border. The thickness will pulse around this value.
    
    pulse_range : int, optional, default: 6
        The range of thickness variation as the border pulses.
    
    frequency : float, optional, default: 0.4
        The frequency at which the border pulses, represented in Hz (cycles per second).
    
    Returns:
    --------
    numpy.ndarray
        The frame with the visual alert applied (pulsing red border and alert text).
    
    Notes:
    ------
    The visual alert consists of two parts:
        1. A pulsing red border around the frame that changes in thickness and color.
        2. A centered "ALERT!" text that pulsates in thickness.
    """
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
    """
    Trigger a sound alert by playing a beep sound. On Windows, it uses the
    built-in `winsound` module, while on Unix-like systems, it attempts to
    play a `.wav` file using the `simpleaudio` library and also triggers
    the system beep sound.
    
    Notes:
    ------
    For Unix-like systems, ensure that the `beep.wav` sound file is located
    in the appropriate directory (relative to the script).
    """
    if sys.platform == "win32":
        # For Windows systems
        import winsound
        winsound.Beep(1000, 500)
    else:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sound = os.path.join(BASE_DIR, "media", "beep.wav")

        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(sound)  # Must be .wav
        # For Unix-like systems, the beep command
        os.system('echo -e "\a"')
        wave_obj.play()  # Non-blocking
