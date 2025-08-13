"""
Main entry point for the Virtual Fabricator application.
"""
import cv2
import time
import threading
from src.ui import VirtualFabricatorUI
from src.gestures import HandTracker, GestureType, draw_landmarks

class VirtualFabricatorApp:
    def __init__(self):
        # Initialize the UI
        self.ui = VirtualFabricatorUI()
        
        # Initialize the hand tracker
        self.hand_tracker = HandTracker()
        
        # Thread control
        self.running = False
        self.cap = None
        
    def start_webcam(self):
        """Start the webcam capture in a separate thread."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.ui.log("Error: Could not open webcam")
            return False
            
        self.running = True
        self.ui.log("Webcam started")
        return True
        
    def process_frames(self):
        """Process webcam frames and detect gestures."""
        prev_time = 0
        fps = 0
        
        while self.running and self.cap.isOpened():
            # Read frame from webcam
            success, frame = self.cap.read()
            if not success:
                self.ui.log("Failed to capture frame from webcam")
                break
                
            # Calculate FPS
            current_time = time.time()
            if prev_time > 0:
                fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
            prev_time = current_time
            
            # Process frame for hand tracking
            hand_landmarks = self.hand_tracker.process_frame(frame)
            
            # Get the latest gesture (non-blocking)
            gesture_event = self.hand_tracker.get_latest_gesture()
            
            # If we have hand landmarks but no gesture, create a default gesture
            if hand_landmarks and not gesture_event:
                # Create a default gesture with the hand position
                index_tip = hand_landmarks.landmarks[8]  # Index finger tip
                gesture_event = type('GestureEvent', (), {
                    'gesture_type': GestureType.INDEX_POINT,
                    'gesture_data': {
                        'position': (index_tip[0], index_tip[1]),  # Normalized coordinates
                        'confidence': 0.9
                    },
                    'timestamp': time.time()
                })
            
            # Ensure UI receives an update every frame (gesture or None)
            if gesture_event:
                # Make sure we have position data
                if not hasattr(gesture_event, 'gesture_data') or 'position' not in gesture_event.gesture_data:
                    if hand_landmarks:
                        # Use index finger tip as position
                        index_tip = hand_landmarks.landmarks[8]  # Index finger tip
                        if not hasattr(gesture_event, 'gesture_data'):
                            gesture_event.gesture_data = {}
                        gesture_event.gesture_data['position'] = (index_tip[0], index_tip[1])
                # Enqueue the gesture to the UI (thread-safe)
                self.ui.enqueue_gesture_from_callback(lambda: gesture_event)
            else:
                # Send None to hide the cursor when no gesture/hand is present
                self.ui.enqueue_gesture_from_callback(lambda: None)
            
            # Draw landmarks on the frame (for debugging)
            if hand_landmarks:
                frame = draw_landmarks(frame, hand_landmarks.landmarks)
            
            # Display FPS on frame (for debugging)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame (for debugging)
            cv2.imshow('Hand Tracking', frame)
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def run(self):
        """Run the application."""
        try:
            # Start webcam
            if not self.start_webcam():
                return
                
            # Start frame processing in a separate thread
            process_thread = threading.Thread(target=self.process_frames)
            process_thread.daemon = True
            process_thread.start()
            
            # Start the UI main loop
            self.ui.run()
            
        except Exception as e:
            self.ui.log(f"Error: {str(e)}")
            
        finally:
            # Cleanup
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.hand_tracker.release()

if __name__ == "__main__":
    # Create and run the application
    app = VirtualFabricatorApp()
    app.run()
