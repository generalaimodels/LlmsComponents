import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

import numpy as np
import cv2
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ultralytics import YOLO
import supervision as sv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "yolov8x.pt"
SUPPORTED_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
OUTPUT_DIR = "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def select_video_path() -> Optional[str]:
    """
    Opens a file dialog for the user to select a video file.
    
    Returns:
        The path to the selected video file or None if cancelled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    root.destroy()
    if video_path:
        logger.info(f"Selected video path: {video_path}")
    else:
        logger.info("No video selected.")
    return video_path or None


def initialize_model(model_path: str) -> YOLO:
    """
    Initializes the YOLO model.
    
    Args:
        model_path: Path to the YOLO model weights.
    
    Returns:
        An instance of the YOLO model.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = YOLO(model_path)
    model.fuse()
    logger.info("YOLO model initialized and fused successfully.")
    return model


def create_annotators() -> Tuple[sv.BoxAnnotator, sv.TraceAnnotator, sv.LineZoneAnnotator]:
    """
    Creates annotators for bounding boxes, traces, and line zones.
    
    Returns:
        A tuple containing box_annotator, trace_annotator, and line_zone_annotator.
    """
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
    logger.info("Annotators created successfully.")
    return box_annotator, trace_annotator, line_zone_annotator


def load_video_info(source_path: str) -> sv.VideoInfo:
    """
    Loads video information from the source path.
    
    Args:
        source_path: Path to the video file.
    
    Returns:
        An instance of VideoInfo.
    """
    video_info = sv.VideoInfo.from_video_path(source_path)
    logger.info(f"Video Info: {video_info}")
    return video_info


def setup_line_zone(video_info: sv.VideoInfo) -> sv.LineZone:
    """
    Sets up the line zone for counting vehicles.
    
    Args:
        video_info: Information about the video.
    
    Returns:
        An instance of LineZone.
    """
    # Position the counting line at 80% of the frame height
    y_position = int(video_info.height * 0.8)
    line_start = sv.Point(x=50, y=y_position)
    line_end = sv.Point(x=video_info.width - 50, y=y_position)
    line_zone = sv.LineZone(start=line_start, end=line_end)
    logger.info(f"LineZone set from {line_start} to {line_end}.")
    return line_zone


def visualize_counts(counts: Dict[str, int], output_path: str) -> None:
    """
    Visualizes vehicle counts using Plotly and saves the figure.
    
    Args:
        counts: Dictionary with vehicle types as keys and counts as values.
        output_path: Path to save the plotly figure.
    """
    vehicle_types = list(counts.keys())
    vehicle_counts = list(counts.values())

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Vehicle Counts", "Count Distribution"))

    # Bar Chart
    fig.add_trace(
        go.Bar(x=vehicle_types, y=vehicle_counts, name="Counts"),
        row=1, col=1
    )

    # Pie Chart
    fig.add_trace(
        go.Pie(labels=vehicle_types, values=vehicle_counts, name="Distribution"),
        row=1, col=2
    )

    fig.update_layout(title_text="Vehicle Monitoring System", legend_title="Vehicle Types")

    # Save the figure as HTML
    fig.write_html(output_path)
    logger.info(f"Plotly figure saved to {output_path}.")


def process_video(
    model: YOLO,
    source_path: str,
    output_video_path: str,
    counts_output_path: str
) -> None:
    """
    Processes the video for vehicle detection and counting.
    
    Args:
        model: The YOLO model instance.
        source_path: Path to the input video.
        output_video_path: Path to save the annotated video.
        counts_output_path: Path to save the counts visualization.
    """
    try:
        video_info = load_video_info(source_path)

        byte_tracker = sv.ByteTrack(
            track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=video_info.fps
        )
        line_zone = setup_line_zone(video_info)
        box_annotator, trace_annotator, line_zone_annotator = create_annotators()
        generator = sv.get_video_frames_generator(source_path)

        counts = {classname: 0 for classname in SUPPORTED_CLASSES.values()}
        logged_track_ids: Dict[int, str] = {}

        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            nonlocal counts, logged_track_ids
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[np.isin(detections.class_id, list(SUPPORTED_CLASSES.keys()))]

            detections = byte_tracker.update_with_detections(detections)

            labels = [
                f"#{tracker_id} {SUPPORTED_CLASSES.get(class_id, 'Unknown')} {confidence:0.2f}"
                for confidence, class_id, tracker_id in zip(
                    detections.confidence, detections.class_id, detections.track_id
                )
            ]

            annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            # Update counts
            line_zone.trigger(detections)
            for zone in line_zone.triggered_zones:
                for detection in zone.object_ids:
                    if detection.id not in logged_track_ids:
                        class_id = detection.class_id
                        class_name = SUPPORTED_CLASSES.get(class_id, "Unknown")
                        counts[class_name] += 1
                        logged_track_ids[detection.id] = class_name

            annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
            return annotated_frame

        # Process video
        sv.process_video(
            source_path=source_path,
            target_path=output_video_path,
            callback=callback
        )

        # Visualize and save counts
        visualize_counts(counts, counts_output_path)
        logger.info("Video processing completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during video processing: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")


def start_processing(model: YOLO, video_path: str) -> None:
    """
    Initiates the video processing in a separate thread.
    
    Args:
        model: The YOLO model instance.
        video_path: Path to the input video.
    """
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            messagebox.showerror("Error", "Selected video file does not exist.")
            return

        output_video_path = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
        counts_output_path = os.path.join(OUTPUT_DIR, "vehicle_counts.html")

        processing_thread = threading.Thread(
            target=process_video,
            args=(model, video_path, output_video_path, counts_output_path),
            daemon=True
        )
        processing_thread.start()
        logger.info("Video processing started in a separate thread.")

        # Wait for processing to complete
        processing_thread.join()
        messagebox.showinfo("Success", f"Video processed successfully!\nOutputs saved in '{OUTPUT_DIR}'.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")


def create_gui(model: YOLO) -> None:
    """
    Creates the GUI for the application.
    
    Args:
        model: The YOLO model instance.
    """
    root = tk.Tk()
    root.title("Vehicle Monitoring System")
    root.geometry("400x200")
    root.resizable(False, False)

    def on_select_video():
        video_path = select_video_path()
        if video_path:
            start_processing(model, video_path)

    # Create widgets
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(expand=True)

    title = tk.Label(frame, text="Vehicle Monitoring System", font=("Helvetica", 16))
    title.pack(pady=10)

    select_button = tk.Button(
        frame,
        text="Select Video",
        command=on_select_video,
        width=20,
        height=2
    )
    select_button.pack(pady=20)

    root.mainloop()


def main() -> None:
    """
    Main function to run the application.
    """
    try:
        logger.info("Initializing the application...")
        model = initialize_model(MODEL_PATH)
        create_gui(model)
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()