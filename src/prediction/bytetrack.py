import supervision as sv
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class OpticalFlowConfig:
    """Configuration for optical flow parameters."""
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    flow_threshold: float = 0.5  # Threshold for considering flow significant
    iou_threshold: float = 0.3   # IOU threshold for merging predictions
    confidence_threshold: float = 0.5  # Threshold for when to use flow vs detections


class OpticalFlowByteTrack:
    """
    Enhances ByteTrack with optical flow to improve tracking through occlusions.
    Compatible with Supervision's ByteTrack interface.
    """
    
    def __init__(
        self, 
        tracker_config: Optional[Dict] = None,
        flow_config: Optional[OpticalFlowConfig] = None
    ):
        """
        Initialize the tracker.
        
        Args:
            tracker_config: Configuration for ByteTrack
            flow_config: Configuration for optical flow
        """
        # Initialize ByteTrack with provided config or defaults
        self.tracker = sv.ByteTrack(**(tracker_config or {}))
        
        # Initialize optical flow configuration
        self.flow_config = flow_config or OpticalFlowConfig()
        
        # Initialize tracking state
        self.prev_frame = None
        self.prev_tracks = None
        self.track_history = {}  # Store history of each track's positions
        
    def update_with_detections(self, detections: sv.Detections, frame) -> sv.Detections:
        """
        Update tracks with detections, enhanced with optical flow.
        Compatible with Supervision's update_with_detections method.
        
        Args:
            detections: Detections from current frame
            frame: Current video frame
            
        Returns:
            sv.Detections with tracking IDs
        """
        # Convert frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply optical flow enhancement if we have previous data
        if self.prev_frame is not None and self.prev_tracks is not None and len(self.prev_tracks) > 0:
            # Calculate optical flow between previous and current frame
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, 
                gray, 
                None, 
                self.flow_config.pyr_scale,
                self.flow_config.levels,
                self.flow_config.winsize,
                self.flow_config.iterations,
                self.flow_config.poly_n,
                self.flow_config.poly_sigma,
                self.flow_config.flags
            )
            
            # Predict new positions for previously tracked objects using flow
            flow_detections = self._predict_positions_from_flow(flow)
            
            # Merge flow-based predictions with current detections
            if flow_detections is not None:
                detections = self._merge_detections(detections, flow_detections)
        
        # Update tracking with ByteTrack
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Update track history
        self._update_track_history(tracked_detections)
        
        # Store current frame and tracks for next iteration
        self.prev_frame = gray
        self.prev_tracks = tracked_detections
        
        return tracked_detections
        
    def _predict_positions_from_flow(self, flow) -> Optional[sv.Detections]:
        """
        Use optical flow to predict new positions of previously tracked objects.
        
        Args:
            flow: Optical flow field
            
        Returns:
            sv.Detections object with predicted positions
        """
        if len(self.prev_tracks) == 0:
            return None
            
        # Get previously tracked bounding boxes and tracking IDs
        prev_boxes = self.prev_tracks.xyxy
        prev_confidences = self.prev_tracks.confidence
        prev_class_ids = self.prev_tracks.class_id
        prev_tracker_ids = self.prev_tracks.tracker_id
        
        predicted_boxes = []
        predicted_confidences = []
        predicted_class_ids = []
        predicted_tracker_ids = []
        
        # For each previous box, calculate its new position based on flow
        for i, box in enumerate(prev_boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Constrain to image boundaries
            height, width = flow.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width-1, x2), min(height-1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Calculate average flow in the box region
            region_flow = flow[y1:y2, x1:x2]
            if region_flow.size == 0:
                continue
                
            avg_flow_x = np.mean(region_flow[..., 0])
            avg_flow_y = np.mean(region_flow[..., 1])
            
            # Only consider significant flow
            flow_magnitude = np.sqrt(avg_flow_x**2 + avg_flow_y**2)
            if flow_magnitude < self.flow_config.flow_threshold:
                avg_flow_x = 0
                avg_flow_y = 0
            
            # Apply flow to move the box
            new_box = [
                x1 + avg_flow_x,
                y1 + avg_flow_y,
                x2 + avg_flow_x,
                y2 + avg_flow_y
            ]
            
            # Constrain predicted box to image boundaries
            new_box[0] = max(0, new_box[0])
            new_box[1] = max(0, new_box[1])
            new_box[2] = min(width, new_box[2])
            new_box[3] = min(height, new_box[3])
            
            # Ensure box has valid dimensions
            if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
                continue
                
            predicted_boxes.append(new_box)
            
            # Reduce confidence for flow-based predictions
            predicted_confidences.append(prev_confidences[i] * 0.8)  # Reduce confidence a bit
            predicted_class_ids.append(prev_class_ids[i])
            predicted_tracker_ids.append(prev_tracker_ids[i])
        
        if not predicted_boxes:
            return None
            
        # Create Detections object from predictions
        return sv.Detections(
            xyxy=np.array(predicted_boxes),
            confidence=np.array(predicted_confidences),
            class_id=np.array(predicted_class_ids),
            tracker_id=np.array(predicted_tracker_ids)
        )
        
    def _merge_detections(self, detector_dets: sv.Detections, flow_dets: sv.Detections) -> sv.Detections:
        """
        Merge detector detections with flow-based predictions.
        
        Args:
            detector_dets: Detections from object detector
            flow_dets: Detections predicted by optical flow
            
        Returns:
            Combined sv.Detections object
        """
        if detector_dets is None or len(detector_dets) == 0:
            return flow_dets
        if flow_dets is None or len(flow_dets) == 0:
            return detector_dets
            
        # Calculate IoU between detector boxes and flow prediction boxes
        detector_boxes = detector_dets.xyxy
        flow_boxes = flow_dets.xyxy
        
        merged_boxes = []
        merged_confidences = []
        merged_class_ids = []
        merged_tracker_ids = []
        
        # Track which flow detections have been merged
        flow_used = [False] * len(flow_dets)
        
        # First, process detector detections and match with flow predictions
        for i, det_box in enumerate(detector_boxes):
            best_iou = 0
            best_match = -1
            
            # Find best matching flow prediction for this detection
            for j, flow_box in enumerate(flow_boxes):
                if flow_dets.tracker_id is None:
                    continue
                    
                iou = self._calculate_iou(det_box, flow_box)
                if iou > best_iou and iou > self.flow_config.iou_threshold:
                    best_iou = iou
                    best_match = j
            
            # If we found a match with good IoU
            if best_match >= 0:
                flow_used[best_match] = True
                
                # If detector confidence is higher, use detector box but keep tracker ID
                if detector_dets.confidence[i] >= self.flow_config.confidence_threshold:
                    merged_boxes.append(detector_dets.xyxy[i])
                    merged_confidences.append(detector_dets.confidence[i])
                    merged_class_ids.append(detector_dets.class_id[i])
                    merged_tracker_ids.append(flow_dets.tracker_id[best_match])
                else:
                    # Otherwise use flow prediction
                    merged_boxes.append(flow_dets.xyxy[best_match])
                    merged_confidences.append(flow_dets.confidence[best_match])
                    merged_class_ids.append(flow_dets.class_id[best_match])
                    merged_tracker_ids.append(flow_dets.tracker_id[best_match])
            else:
                # No match, just add the detection
                merged_boxes.append(detector_dets.xyxy[i])
                merged_confidences.append(detector_dets.confidence[i])
                merged_class_ids.append(detector_dets.class_id[i])
                if detector_dets.tracker_id is not None and i < len(detector_dets.tracker_id):
                    merged_tracker_ids.append(detector_dets.tracker_id[i])
                else:
                    merged_tracker_ids.append(None)
        
        # Add unmatched flow predictions (likely occluded objects)
        for i, used in enumerate(flow_used):
            if not used:
                merged_boxes.append(flow_dets.xyxy[i])
                merged_confidences.append(flow_dets.confidence[i])
                merged_class_ids.append(flow_dets.class_id[i])
                merged_tracker_ids.append(flow_dets.tracker_id[i])
        
        if not merged_boxes:
            return detector_dets
            
        # Create combined Detections object
        return sv.Detections(
            xyxy=np.array(merged_boxes),
            confidence=np.array(merged_confidences),
            class_id=np.array(merged_class_ids),
            tracker_id=np.array(merged_tracker_ids) if merged_tracker_ids else None
        )
    
    def _calculate_iou(self, box1, box2) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0
    
    def _update_track_history(self, detections: sv.Detections):
        """
        Update history of tracked objects.
        
        Args:
            detections: Current tracked detections
        """
        if detections.tracker_id is None or len(detections) == 0:
            return
            
        # Update history for each track
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is None:
                continue
                
            center_x = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
            center_y = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                
            self.track_history[track_id].append((center_x, center_y))
            
            # Keep history limited to prevent memory issues
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
