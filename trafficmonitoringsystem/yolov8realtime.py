import sys
import cv2
import logging
from typing import List, Union
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
import plotly.graph_objs as go
import plotly.io as pio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants for traffic density thresholds
LOW_THRESHOLD = 10     # Less than 10 vehicles per frame
MEDIUM_THRESHOLD = 30  # 10 to 30 vehicles per frame

# Vehicle classes based on YOLOv8's COCO dataset
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorbike', 'bicycle'}


def create_output_folder(folder_name: str) -> Path:
    """
    Create an output folder if it doesn't exist.

    :param folder_name: Name of the folder to create.
    :return: Path object of the created folder.
    """
    try:
        output_path = Path(folder_name)
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output folder is set to: {output_path.resolve()}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to create output folder '{folder_name}': {e}")
        sys.exit(1)


def classify_traffic_density(vehicle_count: int) -> str:
    """
    Classify traffic density based on the number of vehicles detected.

    :param vehicle_count: Number of vehicles detected in the frame.
    :return: Traffic density category as a string.
    """
    if vehicle_count < LOW_THRESHOLD:
        return 'Low Traffic'
    elif LOW_THRESHOLD <= vehicle_count < MEDIUM_THRESHOLD:
        return 'Medium Traffic'
    else:
        return 'High Traffic'


def generate_html_report(timestamps: List[datetime],
                        vehicle_counts: List[int],
                        output_folder: Path) -> None:
    """
    Generate an HTML report with traffic density visualization using Plotly.

    :param timestamps: List of timestamps corresponding to each frame.
    :param vehicle_counts: List of vehicle counts per frame.
    :param output_folder: Path object where the report will be saved.
    """
    try:
        if not timestamps or not vehicle_counts:
            logging.warning("No data available to generate report.")
            return

        # Convert timestamps to string for better readability in the plot
        time_strings = [ts.strftime("%H:%M:%S") for ts in timestamps]

        # Create a line chart for vehicle counts over time
        trace = go.Scatter(
            x=time_strings,
            y=vehicle_counts,
            mode='lines+markers',
            name='Vehicle Count',
            line=dict(color='blue')
        )

        layout = go.Layout(
            title='Traffic Density Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Number of Vehicles'),
            hovermode='closest'
        )

        fig = go.Figure(data=[trace], layout=layout)

        # Define the report file path with timestamp
        report_filename = f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = output_folder / report_filename

        # Save the plotly figure as an HTML file
        pio.write_html(fig, file=report_path, auto_open=False)
        logging.info(f"HTML report generated at: {report_path.resolve()}")

    except Exception as e:
        logging.error(f"Failed to generate HTML report: {e}")


def load_yolo_model(model_path: str = 'yolov8n.pt') -> YOLO:
    """
    Load the YOLOv8 model.

    :param model_path: Path to the YOLOv8 weights file.
    :return: Loaded YOLO model.
    """
    try:
        model = YOLO(model_path)
        logging.info("YOLOv8 model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 model: {e}")
        sys.exit(1)


def process_video_stream(source: Union[str, int],
                        model: YOLO,
                        output_folder: Path) -> None:
    """
    Process the video stream, perform object detection, and collect traffic metrics.

    :param source: Video source (0 for webcam or path to video file).
    :param model: Loaded YOLO model for object detection.
    :param output_folder: Path object where the report will be saved.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error(f"Cannot open video source: {source}")
            sys.exit(1)
        else:
            logging.info(f"Video source '{source}' opened successfully.")

        timestamps: List[datetime] = []
        vehicle_counts: List[int] = []

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("No frame received. Exiting...")
                break

            timestamp = datetime.now()
            timestamps.append(timestamp)

            frame_number += 1

            # Perform object detection
            results = model(frame, verbose=False)

            # Extract detected classes and count vehicles
            detections = results[0].boxes
            vehicle_count = 0
            for det in detections:
                cls_id = int(det.cls[0])
                cls_name = model.names.get(cls_id, '')
                if cls_name in VEHICLE_CLASSES:
                    vehicle_count += 1
                    # Draw bounding box
                    try:
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
                        )
                    except Exception as bbox_exception:
                        logging.error(f"Error drawing bounding box: {bbox_exception}")

            vehicle_counts.append(vehicle_count)
            traffic_density = classify_traffic_density(vehicle_count)

            # Display traffic density on the frame
            cv2.putText(
                frame, f'Traffic: {traffic_density}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

            # Display the resulting frame
            cv2.imshow('Real-Time Traffic Monitoring', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit signal received. Stopping video processing...")
                break

            # Logging every 100 frames to avoid excessive log entries
            if frame_number % 100 == 0:
                logging.info(f"Processed {frame_number} frames.")

        # Release the capture and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        # Generate the HTML report
        generate_html_report(timestamps, vehicle_counts, output_folder)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Exiting gracefully...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        generate_html_report(timestamps, vehicle_counts, output_folder)
    except Exception as e:
        logging.error(f"An error occurred during video processing: {e}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(1)


def get_user_choice() -> Union[str, int]:
    """
    Prompt the user to choose between uploading a video or using the webcam.

    :return: Video source (file path or webcam index).
    """
    while True:
        print("\nSelect Video Source:")
        print("1. Upload a video file")
        print("2. Use webcam for real-time recording")
        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == '1':
            file_path = input("Enter the path to the video file: ").strip()
            if Path(file_path).is_file():
                logging.info(f"Selected video file: {file_path}")
                return file_path
            else:
                logging.error(f"File not found: {file_path}")
        elif choice == '2':
            logging.info("Selected webcam for real-time recording.")
            return 0  # Typically, 0 is the default webcam index
        else:
            logging.error("Invalid choice. Please enter 1 or 2.")


def main() -> None:
    """
    Main function to set up and run the traffic monitoring system based on user choice.
    """
    try:
        # Get user choice for video source
        video_source = get_user_choice()

        # Define the output folder for reports
        output_folder = create_output_folder('traffic_reports')

        # Load YOLOv8 model
        model = load_yolo_model()

        # Start processing the video stream
        process_video_stream(video_source, model, output_folder)

    except Exception as e:
        logging.error(f"An unexpected error occurred in the main function: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()