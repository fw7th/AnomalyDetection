"""
time.py

Used to calculate time a detected object spent in zone.

Author: fw7th
Date: 2025-03-26
"""

from typing import Dict  # Dict for type hinting.

from datetime import datetime

import numpy as np

class ClockBasedTimer:
    """
    Timer using spawn time of frames. 

    Attributes
    ----------
    track_id2start_time : dict
        Dictionary which stores the tracking id as the key and the time an object spawn in a video as its value.

    Methods
    -------
    tick(Detections):
        Method for the time calculation.
    """
    def __init__(self):
        """
        Constructor for ClockBasedTimer.
        """
        self.track_id2start_time : Dict[int, datetime] = {}

    def tick(self, Detections):
        """
        Starts the timer and calculates time spent.

        Params
        ------
        Detections : sv.Detections
            The detections from the detection algorithm.
        """
        current_time = datetime.now()
        times = []

        for tracker_id in Detections.tracker_id:
            self.track_id2start_time.setdefault(tracker_id, current_time)

            start_time = self.track_id2start_time[tracker_id]
            time_duration = (current_time - start_time).total_seconds()
            times.append(time_duration)

        return np.array(times)
