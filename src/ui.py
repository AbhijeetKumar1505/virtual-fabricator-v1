import dearpygui.dearpygui as dpg
import numpy as np
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
            # Main layout - using a table to create the desired layout
            with dpg.table(
                header_row=False,
                borders_innerV=True,
                borders_outerV=True,
                borders_innerH=True,
                borders_outerH=True,
                width=-1,
                height=-1
            ):
                # Define columns: left toolbar, main area
                dpg.add_table_column()
                dpg.add_table_column()
                
                # Add rows: top (toolbar + viewport), bottom (log panel)
                with dpg.table_row(height=0.8):
                    # Left toolbar
                    with dpg.table_cell():
                        self.setup_toolbar()
                    
                    # Main viewport area (3D canvas placeholder)
                    with dpg.table_cell():
                        self.setup_viewport()
                
                # Bottom panel for logs
                with dpg.table_row(height=0.2):
                    with dpg.table_cell(span_columns=True):
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
            
            # Placeholder for 3D view (using a drawlist for 2D rendering in MVP)
            with dpg.drawlist(
                width=-1,
                height=-1,
                tag="viewport_canvas"
            ):
                # Draw a simple grid as a placeholder
                self.draw_placeholder_grid()
                
                # Add a simple object in the center
                center_x, center_y = 400, 300
                dpg.draw_circle(
                    (center_x, center_y),
                    50,
                    color=(100, 200, 255),
                    fill=(100, 200, 255, 50),
                    thickness=2,
                    tag="viewport_object"
                )
    
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
                thickness=1
            )
            
        for y in range(0, height, grid_size):
            dpg.draw_line(
                (0, y),
                (width, y),
                color=(50, 50, 50, 100),
                thickness=1
            )
    
    def setup_log_panel(self):
        """Set up the bottom panel for logs and outputs."""
        with dpg.group(horizontal=False):
            dpg.add_text("Logs & Outputs")
            dpg.add_separator()
            
            # Log output area
            with dpg.child_window(
                height=-30,  # Leave space for the bin button
                border=False
            ):
                dpg.add_text("System ready. Waiting for input...", tag="log_output")
            
            # Bin/trash icon in bottom-right
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-30)  # Push to the right
                dpg.add_button(
                    label="🗑️",
                    width=25,
                    height=25,
                    callback=self.clear_logs,
                    tag="trash_button"
                )
    
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
        
        # Auto-scroll to bottom
        if dpg.does_item_exist("log_output"):
            dpg.set_y_scroll("log_output", -1.0)
    
    def clear_logs(self):
        """Clear the log panel."""
        self.logs = []
        dpg.set_value("log_output", "Logs cleared.")
    
    def process_gesture_queue(self, get_gesture_callback: Callable[[], Any]):
        """Process queued gesture events.
        
        Args:
            get_gesture_callback: A function that returns the next gesture event or None
        """
        gesture = get_gesture_callback()
        if gesture:
            self.log(f"Gesture detected: {gesture.gesture_type.name}")
            
            # Handle different gesture types
            if gesture.gesture_type == GestureType.INDEX_POINT:
                self.handle_index_point(gesture)
            elif gesture.gesture_type == GestureType.PINCH:
                self.handle_pinch(gesture)
            # Add more gesture handlers as needed
    
    def handle_index_point(self, gesture):
        """Handle index point gesture."""
        # In a real implementation, this would update the UI based on the gesture
        self.log(f"Index point at {gesture.gesture_data.get('position', 'unknown')}")
    
    def handle_pinch(self, gesture):
        """Handle pinch gesture."""
        # In a real implementation, this would update the UI based on the gesture
        distance = gesture.gesture_data.get('pinch_distance', 0)
        self.log(f"Pinch detected (distance: {distance:.3f})")
    
    def run(self):
        """Run the main UI loop."""
        while dpg.is_dearpygui_running():
            # Here you would typically get gesture events from your gesture detector
            # For now, we'll just process a dummy callback that returns None
            self.process_gesture_queue(lambda: None)
            dpg.render_dearpygui_frame()
        
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