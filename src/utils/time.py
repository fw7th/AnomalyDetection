from typing import Dict

from datetime import datetime

import numpy as np

class ClockBasedTimer:
    def __init__(self):
        self.track_id2start_time : Dict[int, datetime] = {}

    def tick(self, Detections):
        current_time = datetime.now()
        times = []

        for tracker_id in Detections.tracker_id:
            self.track_id2start_time.setdefault(tracker_id, current_time)

            start_time = self.track_id2start_time[tracker_id]
            time_duration = (current_time - start_time).total_seconds()
            times.append(time_duration)

        return np.array(times)
