import cv2
import mediapipe as mp
import time
import queue
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from enum import Enum, auto
from dataclasses import dataclass, field
from collections import deque

class GestureType(Enum):
    NONE = auto()
    INDEX_POINT = auto()
    PINCH = auto()
    DOUBLE_POINT = auto()
    FIST = auto()
    OPEN_PALM = auto()
    HOLD = auto()

@dataclass
class GestureEvent:
    """Container for gesture detection events."""
    gesture_type: GestureType
    timestamp: float
    data: dict = field(default_factory=dict)  # Additional gesture-specific data

@dataclass
class HandLandmarks:
    """Container for hand landmarks with timestamp and gesture detection."""
    landmarks: np.ndarray  # Shape: (21, 3) - x, y, z coordinates for 21 landmarks
    timestamp: float
    handness: float  # Confidence of hand detection (0-1)
    gesture: GestureType = GestureType.NONE
    gesture_confidence: float = 0.0
    gesture_data: dict = field(default_factory=dict)  # Additional gesture data


def to_pixel_coords(landmark: np.ndarray, width: int, height: int) -> Tuple[int, int]:
    """Convert normalized landmark coordinates to pixel coordinates.
    
    Args:
        landmark: Normalized landmark coordinates (x, y, z)
        width: Width of the image/frame
        height: Height of the image/frame
        
    Returns:
        Tuple of (x, y) pixel coordinates
    """
    x, y = landmark[0], landmark[1]
    return int(x * width), int(y * height)


def smooth_point(prev: Optional[float], current: float, alpha: float = 0.6) -> float:
    """Apply exponential moving average smoothing to a single coordinate.
    
    Args:
        prev: Previous smoothed value (None for first frame)
        current: Current raw value
        alpha: Smoothing factor (0-1), higher = more smoothing
        
    Returns:
        Smoothed value
    """
    if prev is None:
        return current
    return alpha * prev + (1 - alpha) * current


class HandTracker:
    # Landmark indices for MediaPipe Hands
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    def __init__(self, max_queue_size: int = 10):
        """Initialize the hand tracker.
        
        Args:
            max_queue_size: Maximum number of frames to store in the output queue
        """
        self.mp_hands = mp.solutions.hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.landmarks_queue = queue.Queue(maxsize=max_queue_size)
        self.gesture_events = queue.Queue(maxsize=max_queue_size)
        
        # For smoothing
        self.prev_landmarks = None
        self.smoothing_alpha = 0.6
        
        # For gesture detection
        self.last_index_point_time = 0
        self.last_gesture = GestureType.NONE
        self.gesture_start_time = 0
        self.pinch_threshold = 0.04
        self.hold_threshold = 0.3  # seconds
        self.gesture_history = deque(maxlen=5)  # Track recent gestures for double-point detection
        
    def process_frame(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        """Process a single frame to detect and track hands.
        
        Args:
            frame: Input BGR image
            
        Returns:
            HandLandmarks object if hands detected, None otherwise
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get hand landmarks
        results = self.mp_hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handness = results.multi_handedness[0].classification[0].score
        
        # Extract landmarks and apply smoothing
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
        ])
        
        # Apply smoothing to x,y coordinates
        if self.prev_landmarks is not None:
            for i in range(21):
                landmarks[i, 0] = smooth_point(
                    self.prev_landmarks[i, 0], landmarks[i, 0], self.smoothing_alpha
                )
                landmarks[i, 1] = smooth_point(
                    self.prev_landmarks[i, 1], landmarks[i, 1], self.smoothing_alpha
                )
        
        self.prev_landmarks = landmarks.copy()
        
        # Create and queue the result
        result = HandLandmarks(
            landmarks=landmarks,
            timestamp=time.time(),
            handness=handness
        )
        
        # Non-blocking put
        if not self.landmarks_queue.full():
            try:
                self.landmarks_queue.put_nowait(result)
            except queue.Full:
                pass  # Drop the frame if queue is full
                
        return result
    
    def get_latest_landmarks(self, timeout: float = 0.1) -> Optional[HandLandmarks]:
        """Get the latest hand landmarks from the queue.
        
        Args:
            timeout: Maximum time to wait for new landmarks (seconds)
            
        Returns:
            Latest HandLandmarks or None if queue is empty
        """
        try:
            return self.landmarks_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def release(self):
        """Release resources."""
        self.mp_hands.close()
        
    def __enter__(self):
        return self
        
    def _get_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)

    def _is_finger_extended(self, landmarks: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> bool:
        """Check if a finger is extended based on landmark positions."""
        # Check if the finger is extended (tip is above PIP joint)
        return landmarks[tip, 1] < landmarks[pip, 1] and landmarks[dip, 1] < landmarks[pip, 1]

    def _is_finger_folded(self, landmarks: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> bool:
        """Check if a finger is folded."""
        # Check if the finger is folded (tip is close to the base)
        return landmarks[tip, 1] > landmarks[mcp, 1]

    def _detect_gestures(self, landmarks: np.ndarray, timestamp: float) -> Tuple[GestureType, float, dict]:
        """Detect hand gestures based on landmark positions."""
        # Initialize gesture data
        gesture = GestureType.NONE
        confidence = 0.0
        gesture_data = {}

        # Calculate distances for pinch detection
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        pinch_distance = self._get_distance(thumb_tip, index_tip)
        
        # Check for PINCH gesture
        if pinch_distance < self.pinch_threshold:
            gesture = GestureType.PINCH
            confidence = 1.0 - (pinch_distance / self.pinch_threshold)  # Closer to 1.0 as distance decreases
            gesture_data = {"pinch_distance": pinch_distance}
            
            # Check for zoom (pinch distance change)
            if hasattr(self, 'last_pinch_distance'):
                delta = pinch_distance - self.last_pinch_distance
                gesture_data["pinch_delta"] = delta
            self.last_pinch_distance = pinch_distance
            return gesture, confidence, gesture_data
        
        # Reset pinch tracking if not pinching
        if hasattr(self, 'last_pinch_distance'):
            del self.last_pinch_distance

        # Check for INDEX_POINT gesture
        index_extended = self._is_finger_extended(
            landmarks, self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP
        )
        
        other_fingers_folded = all([
            not self._is_finger_extended(
                landmarks, 
                getattr(self, f"{finger}_MCP"), 
                getattr(self, f"{finger}_PIP"), 
                getattr(self, f"{finger}_DIP"), 
                getattr(self, f"{finger}_TIP")
            ) for finger in ["MIDDLE", "RING", "PINKY"]
        ])

        if index_extended and other_fingers_folded:
            current_time = time.time()
            # Check for DOUBLE_POINT (two index points within 0.5s)
            if (current_time - self.last_index_point_time) < 0.5:
                gesture = GestureType.DOUBLE_POINT
                confidence = 1.0
                self.last_index_point_time = 0  # Reset to prevent multiple double points
            else:
                gesture = GestureType.INDEX_POINT
                confidence = 0.9
                self.last_index_point_time = current_time
            
            # Check if this gesture is being held
            if self.last_gesture == gesture:
                hold_duration = timestamp - self.gesture_start_time
                if hold_duration >= self.hold_threshold:
                    gesture = GestureType.HOLD
                    confidence = min(1.0, hold_duration / self.hold_threshold)
                    gesture_data = {"hold_duration": hold_duration}
            else:
                self.gesture_start_time = timestamp
            
            return gesture, confidence, gesture_data

        # Check for FIST gesture
        # Thumb has different joint names in MediaPipe model
        fingers_folded = []
        
        # Check thumb separately
        thumb_folded = self._is_finger_folded(
            landmarks,
            self.THUMB_CMC,
            self.THUMB_MCP,
            self.THUMB_IP,
            self.THUMB_TIP
        )
        fingers_folded.append(thumb_folded)
        
        # Check other fingers
        for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]:
            folded = self._is_finger_folded(
                landmarks,
                getattr(self, f"{finger}_MCP"),
                getattr(self, f"{finger}_PIP"),
                getattr(self, f"{finger}_DIP"),
                getattr(self, f"{finger}_TIP")
            )
            fingers_folded.append(folded)
        
        all_fingers_folded = all(fingers_folded)
        
        if fingers_folded:
            gesture = GestureType.FIST
            confidence = 0.9
            return gesture, confidence, {}

        # Check for OPEN_PALM gesture
        # Thumb has different joint names in MediaPipe model
        fingers_extended = []
        
        # Check thumb separately
        thumb_extended = self._is_finger_extended(
            landmarks,
            self.THUMB_CMC,
            self.THUMB_MCP,
            self.THUMB_IP,
            self.THUMB_TIP
        )
        fingers_extended.append(thumb_extended)
        
        # Check other fingers
        for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]:
            extended = self._is_finger_extended(
                landmarks,
                getattr(self, f"{finger}_MCP"),
                getattr(self, f"{finger}_PIP"),
                getattr(self, f"{finger}_DIP"),
                getattr(self, f"{finger}_TIP")
            )
            fingers_extended.append(extended)
        
        all_fingers_extended = all(fingers_extended)
        
        if fingers_extended:
            gesture = GestureType.OPEN_PALM
            confidence = 0.9
            return gesture, confidence, {}

        return gesture, confidence, gesture_data

    def process_frame(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        """Process a single frame to detect and track hands.
        
        Args:
            frame: Input BGR image
            
        Returns:
            HandLandmarks object if hands detected, None otherwise
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get hand landmarks
        results = self.mp_hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            # Reset gesture tracking if no hands detected
            if self.last_gesture != GestureType.NONE:
                self.last_gesture = GestureType.NONE
                self.gesture_start_time = 0
            return None
            
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handness = results.multi_handedness[0].classification[0].score
        
        # Extract landmarks and apply smoothing
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
        ])
        
        # Apply smoothing to x,y coordinates
        if self.prev_landmarks is not None:
            for i in range(21):
                landmarks[i, 0] = smooth_point(
                    self.prev_landmarks[i, 0], landmarks[i, 0], self.smoothing_alpha
                )
                landmarks[i, 1] = smooth_point(
                    self.prev_landmarks[i, 1], landmarks[i, 1], self.smoothing_alpha
                )
        
        self.prev_landmarks = landmarks.copy()
        
        # Create the result object
        result = HandLandmarks(
            landmarks=landmarks,
            timestamp=time.time(),
            handness=handness
        )
        
        # Detect gestures
        gesture, confidence, gesture_data = self._detect_gestures(landmarks, result.timestamp)
        
        # Update result with gesture information
        result.gesture = gesture
        result.gesture_confidence = confidence
        result.gesture_data = gesture_data
        
        # Update gesture tracking
        if gesture != self.last_gesture:
            # Gesture changed, emit event
            event = GestureEvent(
                gesture_type=gesture,
                timestamp=result.timestamp,
                data=gesture_data
            )
            if not self.gesture_events.full():
                try:
                    self.gesture_events.put_nowait(event)
                except queue.Full:
                    pass
        
        self.last_gesture = gesture
        
        # Non-blocking put to landmarks queue
        if not self.landmarks_queue.full():
            try:
                self.landmarks_queue.put_nowait(result)
            except queue.Full:
                pass  # Drop the frame if queue is full
                
        return result
    
    def get_latest_gesture(self, timeout: float = 0.1) -> Optional[GestureEvent]:
        """Get the latest gesture event from the queue.
        
        Args:
            timeout: Maximum time to wait for a new gesture event (seconds)
            
        Returns:
            Latest GestureEvent or None if queue is empty
        """
        try:
            return self.gesture_events.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=3):
    """Draw hand landmarks on the image."""
    h, w = image.shape[:2]
    for lm in landmarks:
        x, y = to_pixel_coords(lm, w, h)
        cv2.circle(image, (x, y), radius, color, -1)
    return image

def draw_gesture_info(image, gesture: GestureType, confidence: float, data: dict):
    """Draw gesture information on the image."""
    gesture_text = f"{gesture.name}"
    if confidence > 0:
        gesture_text += f" ({confidence:.1f})"
    
    # Draw gesture name
    cv2.putText(image, gesture_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw additional gesture data
    if gesture == GestureType.PINCH and 'pinch_distance' in data:
        cv2.putText(image, f"Distance: {data['pinch_distance']:.3f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif gesture == GestureType.HOLD and 'hold_duration' in data:
        cv2.putText(image, f"Hold: {data['hold_duration']:.1f}s", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image

def main():
    """Example usage of the HandTracker with gesture detection."""
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    # For FPS calculation
    prev_time = 0
    fps = 0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Calculate FPS
            current_time = time.time()
            if prev_time > 0:
                fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
            prev_time = current_time
            
            # Process the frame
            result = tracker.process_frame(frame)
            
            # Draw landmarks and gesture info if hand detected
            if result is not None:
                # Draw landmarks
                draw_landmarks(frame, result.landmarks)
                
                # Draw gesture info
                draw_gesture_info(
                    frame, 
                    result.gesture,
                    result.gesture_confidence,
                    result.gesture_data
                )
            
            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Check for gesture events
            gesture_event = tracker.get_latest_gesture()
            if gesture_event:
                print(f"Gesture detected: {gesture_event.gesture_type.name}")
                if gesture_event.gesture_type == GestureType.DOUBLE_POINT:
                    print("  → Double point detected!")
                elif gesture_event.gesture_type == GestureType.HOLD:
                    print(f"  → Holding for {gesture_event.data.get('hold_duration', 0):.1f} seconds")
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()


if __name__ == "__main__":
    main()