import dearpygui.dearpygui as dpg
import numpy as np
import queue
from typing import Optional, Callable, Any
from enum import Enum, auto
import time

class UIMode(Enum):
    MOVE = auto()
    ROTATE = auto()
    SCALE = auto()
    IMAGE_GEN = auto()

class VirtualFabricatorUI:
    def __init__(self):
        self.viewport_width = 1200
        self.viewport_height = 800
        self.current_mode: UIMode = UIMode.MOVE
        self.logs = []
        
        # For click detection
        self.last_click_time = 0
        self.click_debounce = 0.3  # seconds
        self.last_gesture_position = (0, 0)
        self.gesture_start_time = 0
        self.selected_object = None
        self.is_holding = False
        self.is_dragging = False
        self.bin_highlight = False
        
        # For viewport objects (example objects)
        self.viewport_objects = [
            {'id': 'obj1', 'x': 400, 'y': 300, 'width': 100, 'height': 100, 'selected': False}
        ]
        
        # For tracking hand position
        self.hand_position = None
        self.hand_visible = False
        
        # Thread-safe queue for gestures coming from other threads
        self._gesture_queue: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=128)
        self._pending_redraw: bool = False

        # Initialize the UI
        self.setup_ui()
        
    def setup_ui(self):
        dpg.create_context()
        
        # Set up the viewport
        dpg.create_viewport(
            title="Virtual Fabricator",
            width=self.viewport_width,
            height=self.viewport_height,
            resizable=True
        )
        
        # Set up the main window
        with dpg.window(tag="Primary Window"):
            # Main layout using groups instead of tables
            with dpg.group(horizontal=True, width=-1, height=-1):
                # Left toolbar (20% width)
                with dpg.child_window(width=200, border=True):
                    self.setup_toolbar()
                
                # Main area (80% width)
                with dpg.group(horizontal=False, width=-1, height=-1):
                    # Viewport area (80% height)
                    with dpg.child_window(height=-120, border=True):
                        self.setup_viewport()
                    
                    # Log panel (20% height, full width)
                    with dpg.child_window(height=100, border=True):
                        self.setup_log_panel()
        
        # Set primary window to fill the viewport
        dpg.set_primary_window("Primary Window", True)
        
        # Set up the viewport
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
    def setup_toolbar(self):
        """Set up the left toolbar with mode buttons."""
        with dpg.group(horizontal=False):
            dpg.add_text("Tools")
            dpg.add_separator()
            
            # Mode selection buttons
            self.add_tool_button("Move", UIMode.MOVE, "Move objects")
            self.add_tool_button("Rotate", UIMode.ROTATE, "Rotate objects")
            self.add_tool_button("Scale", UIMode.SCALE, "Scale objects")
            dpg.add_separator()
            self.add_tool_button("Image Generation", UIMode.IMAGE_GEN, "Generate images")
    
    def add_tool_button(self, label: str, mode: UIMode, tooltip: str = ""):
        """Helper to add a tool button with consistent styling."""
        with dpg.group(horizontal=True):
            dpg.add_radio_button(
                [label],
                callback=lambda s, a, m=mode: self.set_mode(m),
                tag=f"tool_{mode.name.lower()}",
            )
            if tooltip:
                dpg.add_text("(?)", color=(100, 255, 100))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(tooltip)
    
    def setup_viewport(self):
        """Set up the 3D viewport area."""
        with dpg.group(horizontal=False):
            dpg.add_text("3D Viewport")
            dpg.add_separator()
            
            # Main viewport container
            with dpg.group(tag="viewport_container"):
                # Drawlist for the viewport content
                with dpg.drawlist(
                    width=-1,
                    height=-1,
                    tag="viewport_canvas"
                ):
                    # Draw a simple grid as a placeholder
                    self.draw_placeholder_grid()
                    
                    # Add a simple object in the center
                    self.draw_viewport_objects()
                    
                    # Draw hand cursor when visible
                    self.draw_hand_cursor()
                    
                    # Draw fixed bin icon bottom-right
                    self.draw_bin_icon()
    
    def draw_placeholder_grid(self):
        """Draw a simple grid in the viewport."""
        width = dpg.get_item_width("viewport_canvas")
        height = dpg.get_item_height("viewport_canvas")
        grid_size = 50
        
        # Draw grid lines
        for x in range(0, width, grid_size):
            dpg.draw_line(
                (x, 0),
                (x, height),
                color=(50, 50, 50, 100),
                thickness=1,
                parent="viewport_canvas"
            )
            
        for y in range(0, height, grid_size):
            dpg.draw_line(
                (0, y),
                (width, y),
                color=(50, 50, 50, 100),
                thickness=1,
                parent="viewport_canvas"
            )
            
    def draw_hand_cursor(self):
        """Draw a cursor at the hand position."""
        if self.hand_visible and self.hand_position:
            x, y = self.hand_position
            dpg.draw_circle(
                (x, y),
                10,
                color=(255, 0, 0, 200),
                fill=(255, 0, 0, 100),
                thickness=2,
                parent="viewport_canvas",
                tag="hand_cursor"
            )
    
    def draw_viewport_objects(self):
        """Draw all objects in the viewport."""
        for obj in self.viewport_objects:
            color = (0, 255, 0, 200) if obj.get('selected', False) else (100, 200, 255, 100)
            dpg.draw_rectangle(
                (obj['x'], obj['y']),
                (obj['x'] + obj['width'], obj['y'] + obj['height']),
                color=color,
                fill=color,
                thickness=2,
                parent="viewport_canvas",
                tag=f"obj_{obj['id']}"
            )

    def draw_bin_icon(self):
        """Draw a simple bin icon at the bottom-right corner (fixed within the canvas)."""
        try:
            width = dpg.get_item_width("viewport_canvas")
            height = dpg.get_item_height("viewport_canvas")
            if not width or not height:
                width, height = self.viewport_width, self.viewport_height
        except Exception:
            width, height = self.viewport_width, self.viewport_height

        # Bin dimensions and position
        margin = 20
        bin_size = 40
        x1 = max(0, width - margin - bin_size)
        y1 = max(0, height - margin - bin_size)
        x2 = x1 + bin_size
        y2 = y1 + bin_size

        # Background square
        bg_fill = (80, 80, 80, 240) if not self.bin_highlight else (60, 140, 60, 240)
        dpg.draw_rectangle((x1, y1), (x2, y2), color=(180, 180, 180, 255), fill=bg_fill, thickness=1, parent="viewport_canvas", tag="bin_bg")
        # Bin body
        body_margin = 8
        body_color = (230, 230, 230, 255)
        dpg.draw_rectangle((x1 + body_margin, y1 + body_margin + 6), (x2 - body_margin, y2 - body_margin), color=body_color, thickness=2, parent="viewport_canvas", tag="bin_body")
        # Bin lid
        lid_color = (230, 230, 230, 255)
        dpg.draw_line((x1 + body_margin - 2, y1 + body_margin + 4), (x2 - body_margin + 2, y1 + body_margin + 4), color=lid_color, thickness=3, parent="viewport_canvas", tag="bin_lid")

        if self.is_dragging and self.bin_highlight:
            dpg.draw_text((x1 - 140, y1 - 10), "Release to delete", color=(255, 220, 220, 255), size=16, parent="viewport_canvas", tag="bin_hint")

    def get_bin_bbox(self) -> tuple[int, int, int, int]:
        try:
            width = dpg.get_item_width("viewport_canvas")
            height = dpg.get_item_height("viewport_canvas")
            if not width or not height:
                width, height = self.viewport_width, self.viewport_height
        except Exception:
            width, height = self.viewport_width, self.viewport_height

        margin = 20
        bin_size = 40
        x1 = max(0, width - margin - bin_size)
        y1 = max(0, height - margin - bin_size)
        x2 = x1 + bin_size
        y2 = y1 + bin_size
        return x1, y1, x2, y2
    
    def setup_log_panel(self):
        """Set up the bottom panel for logs and outputs."""
        with dpg.group(horizontal=False, width=-1, height=-1):
            # Header with title and clear button
            with dpg.group(horizontal=True):
                dpg.add_text("Logs & Outputs")
                dpg.add_spacer()
                dpg.add_button(
                    label="🗑️",
                    width=25,
                    height=25,
                    callback=self.clear_logs,
                    tag="trash_button"
                )
            
            dpg.add_separator()
            
            # Log output area
            with dpg.child_window(border=False):
                dpg.add_text("System ready. Waiting for input...", tag="log_output")
    
    def set_mode(self, mode: UIMode):
        """Set the current interaction mode."""
        self.current_mode = mode
        self.log(f"Mode set to: {mode.name}")
        
        # Update button states
        for m in UIMode:
            dpg.set_value(f"tool_{m.name.lower()}", m == mode)
    
    def log(self, message: str):
        """Add a message to the log panel."""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.logs.append(log_message)
        
        # Keep log size manageable
        if len(self.logs) > 100:
            self.logs.pop(0)
        
        # Update the log display
        dpg.set_value("log_output", "\n".join(self.logs))
        
        # Auto-scroll to bottom (wrapped in try-except to handle any scroll-related errors)
        try:
            if dpg.does_item_exist("log_output"):
                # Get the parent window of the log output
                parent = dpg.get_item_parent("log_output")
                if parent:
                    # Set the scroll position to the bottom
                    dpg.set_y_scroll(parent, -1.0)
        except Exception as e:
            # Silently ignore scroll errors
            pass
    
    def clear_logs(self):
        """Clear the log panel."""
        self.logs = []
        dpg.set_value("log_output", "Logs cleared.")
    
    def enqueue_gesture(self, gesture: Optional[Any]):
        """Enqueue a gesture event from any thread. Pass None to indicate no gesture/hand hidden."""
        try:
            self._gesture_queue.put_nowait(gesture)
        except queue.Full:
            # Drop if UI is backlogged
            pass

    def _apply_gesture(self, gesture: Any):
        """Apply a single gesture to UI state. Must be called from the UI thread."""
        if not gesture:
            # No gesture, hide hand cursor
            self.hand_visible = False
            self._pending_redraw = True
            return

        # Update hand position if available
        if hasattr(gesture, 'gesture_data') and 'position' in getattr(gesture, 'gesture_data', {}):
            gx, gy = gesture.gesture_data['position']
            # Flip x coordinate for mirror effect (like a mirror)
            gx = 1.0 - gx

            # Use actual canvas size if available
            try:
                canvas_w = dpg.get_item_width("viewport_canvas") or self.viewport_width
                canvas_h = dpg.get_item_height("viewport_canvas") or self.viewport_height
            except Exception:
                canvas_w, canvas_h = self.viewport_width, self.viewport_height

            self.hand_position = (int(gx * canvas_w), int(gy * canvas_h))
            self.hand_visible = True

            # Log position for debugging (rate-limited)
            if hasattr(self, 'last_log_time') and (time.time() - self.last_log_time) > 1.0:
                self.log(f"Hand at: ({self.hand_position[0]}, {self.hand_position[1]})")
                self.last_log_time = time.time()
            elif not hasattr(self, 'last_log_time'):
                self.last_log_time = time.time()

        # Handle different gesture types
        if hasattr(gesture, 'gesture_type'):
            if gesture.gesture_type == GestureType.INDEX_POINT:
                self.handle_index_point(gesture)
            elif gesture.gesture_type == GestureType.PINCH:
                self.handle_pinch(gesture)
            elif gesture.gesture_type == GestureType.DOUBLE_POINT:
                self.handle_double_point(gesture)
            elif gesture.gesture_type == GestureType.OPEN_PALM:
                self.handle_open_palm(gesture)

        self._pending_redraw = True

    def enqueue_gesture_from_callback(self, get_gesture_callback: Callable[[], Any]):
        """Compatibility shim: enqueue one gesture from a callback (may be called from any thread)."""
        try:
            gesture = get_gesture_callback()
            self.enqueue_gesture(gesture)
        except Exception:
            # Ignore errors from producer side
            pass
    
    def update_viewport(self):
        """Update the viewport display."""
        if dpg.does_item_exist("viewport_canvas"):
            # Clear the viewport
            dpg.delete_item("viewport_canvas", children_only=True)
            
            # Redraw all elements
            self.draw_placeholder_grid()
            self.draw_viewport_objects()
            self.draw_hand_cursor()

    def _process_pending_gestures_on_ui_thread(self):
        """Drain and process all queued gestures on the UI thread."""
        processed_any = False
        try:
            while True:
                gesture = self._gesture_queue.get_nowait()
                self._apply_gesture(gesture)
                processed_any = True
        except queue.Empty:
            pass

        if processed_any or self._pending_redraw:
            self.update_viewport()
            self._pending_redraw = False

    def process_gesture_queue(self):
        """Public: process any queued gestures; call once per frame from the UI thread."""
        self._process_pending_gestures_on_ui_thread()
    
    def map_to_viewport(self, x: float, y: float) -> tuple[int, int]:
        """Map normalized coordinates (0..1) to viewport pixels."""
        px = int(x * self.viewport_width)
        py = int(y * self.viewport_height)
        return px, py
        
    def is_point_in_rect(self, point: tuple[float, float], rect: dict) -> bool:
        """Check if a point is inside a rectangle."""
        x, y = point
        return (rect['x'] <= x <= rect['x'] + rect['width'] and 
                rect['y'] <= y <= rect['y'] + rect['height'])
    
    def handle_click(self, x: float, y: float):
        """Handle a click at the given viewport coordinates."""
        current_time = time.time()
        
        # Debounce check
        if current_time - self.last_click_time < self.click_debounce:
            return
            
        self.last_click_time = current_time
        
        # Check if click is on a viewport object
        for obj in self.viewport_objects:
            if self.is_point_in_rect((x, y), obj):
                # Toggle selection
                obj['selected'] = not obj.get('selected', False)
                self.selected_object = obj['id'] if obj['selected'] else None
                self.log(f"Selected object: {self.selected_object}" if obj['selected'] 
                        else "Deselected object")
                return
        
        # Check if click is on a toolbar button
        # Note: You'll need to implement this based on your toolbar layout
        # Example: self.check_toolbar_click(x, y)
        
        # If we get here, the click was in empty space
        if self.selected_object:
            self.log(f"Deselected object {self.selected_object}")
            self.selected_object = None
    
    def handle_hold(self, x: float, y: float, duration: float):
        """Handle a hold gesture at the given coordinates."""
        if duration >= 0.3:
            self.is_holding = True
            # Check if we're holding over the bin area
            # Example: self.check_bin_hold(x, y)
            
    def handle_index_point(self, gesture):
        """Handle index point gesture."""
        try:
            current_time = time.time()
            
            # Get normalized coordinates from gesture data
            if not hasattr(gesture, 'gesture_data') or 'position' not in gesture.gesture_data:
                return
                
            norm_x, norm_y = gesture.gesture_data['position']
            # Flip x coordinate for mirror effect (like a mirror)
            norm_x = 1.0 - norm_x
            x, y = self.map_to_viewport(norm_x, norm_y)
            
            # Update last gesture position
            self.last_gesture_position = (x, y)
            
            # Handle first frame of gesture
            if not hasattr(self, 'gesture_start_time') or self.gesture_start_time == 0:
                self.gesture_start_time = current_time
            
            # Handle hold detection
            hold_duration = current_time - self.gesture_start_time
            self.handle_hold(x, y, hold_duration)
            
            # Check for object selection
            self.check_object_selection(x, y, hold_duration)

            # If dragging, move selected object with fingertip and update bin highlight
            if self.is_dragging and self.selected_object:
                self.move_selected_object_to((x, y))
                self.bin_highlight = self.is_object_center_in_bin(self.get_selected_object())
            
        except Exception as e:
            self.log(f"Error in handle_index_point: {str(e)}")

    def handle_double_point(self, gesture):
        """Begin drag if an object is selected."""
        if not self.selected_object:
            return
        self.is_dragging = True
        self.bin_highlight = False
        self.log("Drag started")

    def handle_open_palm(self, gesture):
        """Drop current drag. Delete if over bin, else place object."""
        if not self.is_dragging or not self.selected_object:
            return
        obj = self.get_selected_object()
        if obj and self.is_object_center_in_bin(obj):
            self.animate_and_delete_selected(obj)
        else:
            self.is_dragging = False
            self.bin_highlight = False
            self.log("Placed object")
    
    def check_object_selection(self, x: int, y: int, hold_duration: float):
        """Check if the hand is pointing at an object and handle selection."""
        if hold_duration >= 0.3:
            for obj in self.viewport_objects:
                if self.is_point_in_rect((x, y), obj):
                    if not obj.get('selected', False):
                        self.select_object(obj['id'])
                    return
            
            # If we get here, no object was selected
            if self.selected_object:
                self.deselect_current()
    
    def handle_pinch(self, gesture):
        """Handle pinch gesture."""
        distance = gesture.gesture_data.get('pinch_distance', 0)
        self.log(f"Pinch detected (distance: {distance:.3f})")
        
        # Handle pinch-to-zoom or other pinch gestures
        if self.selected_object and hasattr(gesture, 'gesture_data'):
            # Example: Scale the selected object based on pinch distance
            # You'll need to implement the actual scaling logic
            pass

    def select_object(self, object_id: str):
        """Select object by id."""
        for obj in self.viewport_objects:
            obj['selected'] = (obj['id'] == object_id)
            if obj['selected']:
                self.selected_object = obj['id']
        self.log(f"Selected object: {self.selected_object}")

    def deselect_current(self):
        """Deselect current selected object."""
        for obj in self.viewport_objects:
            if obj['id'] == self.selected_object:
                obj['selected'] = False
        if self.is_dragging:
            self.is_dragging = False
        self.bin_highlight = False
        self.log("Deselected object")
        self.selected_object = None

    def get_selected_object(self) -> Optional[dict]:
        for obj in self.viewport_objects:
            if obj['id'] == self.selected_object:
                return obj
        return None

    def move_selected_object_to(self, pos: tuple[int, int]):
        obj = self.get_selected_object()
        if not obj:
            return
        x, y = pos
        w, h = obj['width'], obj['height']
        obj['x'] = int(x - w / 2)
        obj['y'] = int(y - h / 2)

    def is_object_center_in_bin(self, obj: dict) -> bool:
        x1, y1, x2, y2 = self.get_bin_bbox()
        cx = obj['x'] + obj['width'] // 2
        cy = obj['y'] + obj['height'] // 2
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def animate_and_delete_selected(self, obj: dict):
        """Animate a brief shrink then delete the selected object."""
        steps = 8
        for i in range(steps):
            scale = max(0, 1.0 - (i + 1) / steps)
            w0, h0 = obj['width'], obj['height']
            cx = obj['x'] + w0 // 2
            cy = obj['y'] + h0 // 2
            obj['width'] = max(1, int(w0 * scale))
            obj['height'] = max(1, int(h0 * scale))
            obj['x'] = int(cx - obj['width'] / 2)
            obj['y'] = int(cy - obj['height'] / 2)
            self.bin_highlight = True
            self.update_viewport()
            dpg.render_dearpygui_frame()
            time.sleep(0.02)
        self.delete_object(obj['id'])
        self.is_dragging = False
        self.bin_highlight = False
        self.log("Deleted object")

    def delete_object(self, object_id: str):
        self.viewport_objects = [o for o in self.viewport_objects if o['id'] != object_id]
        if self.selected_object == object_id:
            self.selected_object = None
        self.update_viewport()
    
    def run(self):
        """Run the main UI loop."""
        last_frame_time = time.time()
        
        while dpg.is_dearpygui_running():
            current_time = time.time()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time
            
            # First process any pending gestures from other threads (non-blocking UI)
            self.process_gesture_queue()

            # Process UI events
            dpg.render_dearpygui_frame()
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
        
        dpg.destroy_context()

# Import the gesture types
# This will work when running as a module (from main.py)
# and will raise an ImportError when running this file directly
# which is handled in the __main__ block below
from .gestures import GestureType

# For testing the UI standalone
if __name__ == "__main__":
    try:
        # Try to import GestureType from the package
        from .gestures import GestureType
    except ImportError:
        # If that fails, try direct import (for running this file directly)
        from gestures import GestureType
    
    # Create and run the UI
    ui = VirtualFabricatorUI()
    ui.run()