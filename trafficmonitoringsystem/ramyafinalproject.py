import sys
import cv2
import logging
from typing import List, Union, Optional, Dict, Tuple
from datetime import datetime
from pathlib import Path
from threading import Thread
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants for traffic density thresholds
LOW_THRESHOLD: int = 5      
MEDIUM_THRESHOLD: int = 10   

# Vehicle classes based on YOLOv8's COCO dataset
VEHICLE_CLASSES: set = {'car', 'truck', 'bus', 'motorbike', 'bicycle'}


class ReportGenerator:
    """
    Generates and saves HTML reports with traffic density and vehicle type visualizations.
    """

    def __init__(self, output_folder: Path) -> None:
        self.output_folder: Path = output_folder

    def generate_html_report(
        self,
        timestamps: List[datetime],
        vehicle_counts: List[int],
        vehicle_type_counts: List[Dict[str, int]]
    ) -> None:
        """
        Generates an HTML report summarizing traffic metrics and vehicle types.

        Args:
            timestamps (List[datetime]): List of timestamps corresponding to each frame.
            vehicle_counts (List[int]): List of total vehicle counts per frame.
            vehicle_type_counts (List[Dict[str, int]]): List of vehicle type counts per frame.
        """
        try:
            if not timestamps or not vehicle_counts:
                logging.warning("No data available to generate report.")
                return

            report_filename: str = f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            report_path: Path = self.output_folder / report_filename

            with report_path.open('w') as report_file:
                report_file.write("<html><head><title>Traffic Report</title></head><body>")
                report_file.write("<h1>Traffic Density Over Time</h1>")
                report_file.write("<table border='1'>")
                report_file.write("<tr><th>Timestamp</th><th>Total Vehicle Count</th>")
                for vehicle_type in sorted(VEHICLE_CLASSES):
                    report_file.write(f"<th>{vehicle_type.capitalize()} Count</th>")
                report_file.write("</tr>")
                for ts, count, type_counts in zip(timestamps, vehicle_counts, vehicle_type_counts):
                    report_file.write(f"<tr><td>{ts.strftime('%H:%M:%S')}</td><td>{count}</td>")
                    for vehicle_type in sorted(VEHICLE_CLASSES):
                        report_file.write(f"<td>{type_counts.get(vehicle_type, 0)}</td>")
                    report_file.write("</tr>")
                report_file.write("</table></body></html>")

            logging.info(f"HTML report generated at: {report_path.resolve()}")

        except Exception as e:
            logging.error(f"Failed to generate HTML report: {e}")


class TrafficMonitor:
    """
    Monitors traffic by processing video streams, detecting vehicles, tracking them, and maintaining traffic metrics.
    """

    def __init__(
        self,
        source: Union[str, int],
        model_path: str = 'yolov8x.pt',
        output_folder: str = 'traffic_reports',
        output_video_path: str = 'output_video.avi'
    ) -> None:
        self.source: Union[str, int] = source
        self.output_folder: Path = Path(output_folder)
        self.output_video_path: Path = Path(output_video_path)
        self.model_path: str = model_path
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.timestamps: List[datetime] = []
        self.vehicle_counts: List[int] = []
        self.vehicle_type_counts: List[Dict[str, int]] = []
        self.report_generator: Optional[ReportGenerator] = None
        self.frame_number: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.legend_ratio: float = 0.2  # Increased to accommodate percentage display
        self.colors: Dict[str, Tuple[int, int, int]] = {
            'Low Traffic': (0, 255, 0),
            'Medium Traffic': (0, 165, 255),
            'High Traffic': (0, 0, 255)
        }
        self.tracked_objects: Dict[int, Dict[str, Union[str, Tuple[int, int, int, int]]]] = {}
        self.next_object_id: int = 0  # ID to assign to the next detected object
        self.lock: bool = False  # Simple lock to prevent concurrent access during cleanup
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.high_traffic_count: int = 0
        self.medium_traffic_count: int = 0
        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize the traffic monitor by setting up the output folder, loading the YOLO model, and initializing the report generator.
        """
        self._create_output_folder()
        self._load_yolo_model()
        self.report_generator = ReportGenerator(self.output_folder)
        self._initialize_video_writer()

    def _create_output_folder(self) -> None:
        """
        Create an output folder if it doesn't exist.
        """
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            logging.info(f"Output folder is set to: {self.output_folder.resolve()}")
        except Exception as e:
            logging.error(f"Failed to create output folder '{self.output_folder}': {e}")
            sys.exit(1)

    def _load_yolo_model(self) -> None:
        """
        Load the YOLOv8 model.
        """
        try:
            self.model = YOLO(self.model_path)
            logging.info("YOLOv8 model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load YOLOv8 model: {e}")
            sys.exit(1)

    def _initialize_video_writer(self) -> None:
        """
        Initialize the VideoWriter to save the annotated output video.
        """
        try:
            # Define the codec and create VideoWriter object
            fourcc: int = cv2.VideoWriter_fourcc(*'XVID')
            # Placeholder frame size; it will be initialized after reading the first frame
            self.video_writer = None
            logging.info("VideoWriter initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize VideoWriter: {e}")
            sys.exit(1)

    @staticmethod
    def classify_traffic_density(vehicle_count: int) -> str:
        """
        Classify traffic density based on the number of vehicles detected.

        Args:
            vehicle_count (int): Number of vehicles detected in the frame.

        Returns:
            str: Traffic density category.
        """
        if vehicle_count < LOW_THRESHOLD:
            return 'Low Traffic'
        elif LOW_THRESHOLD <= vehicle_count < MEDIUM_THRESHOLD:
            return 'Medium Traffic'
        else:
            return 'High Traffic'

    def _process_detections(self, detections) -> Tuple[int, Dict[str, int]]:
        """
        Process YOLO detections, track vehicles, and count vehicle types.

        Args:
            detections: YOLO detections for the current frame.

        Returns:
            Tuple[int, Dict[str, int]]: Number of unique vehicles detected and a dictionary of vehicle type counts.
        """
        vehicle_type_count: Dict[str, int] = defaultdict(int)
        for det in detections:
            try:
                cls_id: int = int(det.cls[0])
                cls_name: str = self.model.names.get(cls_id, '')
                if cls_name in VEHICLE_CLASSES:
                    vehicle_type_count[cls_name] += 1
                    # Draw bounding box and label
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    cv2.rectangle(
                        self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        self.frame, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2
                    )
                    # Assign an ID to the detected object
                    self._assign_object_id(cls_name, (x1, y1, x2, y2))
            except Exception as e:
                logging.error(f"Error processing detection: {e}")
        unique_vehicles = len(self.tracked_objects)
        return unique_vehicles, dict(vehicle_type_count)

    def _assign_object_id(self, cls_name: str, bbox: Tuple[int, int, int, int]) -> None:
        """
        Assign a unique ID to the detected vehicle and update its position.

        Args:
            cls_name (str): Class name of the detected vehicle.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates of the detected vehicle.
        """
        x1, y1, x2, y2 = bbox
        bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        assigned = False
        for obj_id, obj_info in self.tracked_objects.items():
            ox1, oy1, ox2, oy2 = obj_info['bbox']
            o_center = obj_info['center']
            distance = np.linalg.norm(np.array(bbox_center) - np.array(o_center))
            if distance < 50:  # Threshold for considering the same object
                self.tracked_objects[obj_id]['bbox'] = bbox
                self.tracked_objects[obj_id]['center'] = bbox_center
                assigned = True
                break
        if not assigned:
            self.tracked_objects[self.next_object_id] = {
                'class': cls_name,
                'bbox': bbox,
                'center': bbox_center
            }
            self.next_object_id += 1

    def _add_legend_panel(
        self,
        traffic_density: str,
        total_vehicles: int,
        vehicle_type_counts: Dict[str, int]
    ) -> None:
        """
        Adds a legend panel on the right side of the frame displaying vehicle counts, traffic density, and density percentages.

        Args:
            traffic_density (str): The classified traffic density.
            total_vehicles (int): Total number of vehicles detected.
            vehicle_type_counts (Dict[str, int]): Dictionary of vehicle types and their counts.
        """
        try:
            # Calculate legend width based on frame width
            legend_width = int(self.frame_width * self.legend_ratio)
            # Create a blank legend panel
            legend = 255 * np.ones((self.frame_height, legend_width, 3), dtype=np.uint8)

            # Display current total vehicle count
            cv2.putText(
                legend, f"Total Vehicles: {total_vehicles}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
            )

            # Display traffic density
            cv2.putText(
                legend, "Traffic Density:", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
            )
            cv2.rectangle(
                legend, (170, 85), (200, 115),
                self.colors.get(traffic_density, (0, 0, 0)), -1
            )
            cv2.putText(
                legend, traffic_density, (210, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
            )

            # Calculate and display traffic density percentages
            total_classifications = self.high_traffic_count + self.medium_traffic_count + (self.frame_number - self.high_traffic_count - self.medium_traffic_count)
            high_percentage: float = (self.high_traffic_count / total_classifications) * 100 if total_classifications else 0
            medium_percentage: float = (self.medium_traffic_count / total_classifications) * 100 if total_classifications else 0

            cv2.putText(
                legend, f"High Traffic Frames: {high_percentage:.2f}%", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['High Traffic'], 2
            )
            cv2.putText(
                legend, f"Medium Traffic Frames: {medium_percentage:.2f}%", (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['Medium Traffic'], 2
            )

            # Display vehicle type counts
            y_offset = 210
            for vehicle_type in sorted(VEHICLE_CLASSES):
                count = vehicle_type_counts.get(vehicle_type, 0)
                text = f"{vehicle_type.capitalize()}: {count}"
                cv2.putText(
                    legend, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                )
                y_offset += 30

            # Combine the original frame with the legend panel
            self.frame = cv2.hconcat([self.frame, legend])

        except Exception as e:
            logging.error(f"Failed to add legend panel: {e}")

    def process_stream(self) -> None:
        """
        Process the video stream, perform object detection, track vehicles, and collect traffic metrics.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logging.error(f"Cannot open video source: {self.source}")
                sys.exit(1)
            else:
                logging.info(f"Video source '{self.source}' opened successfully.")

            # Retrieve frame dimensions and initialize VideoWriter after first frame
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to read from video source.")
                sys.exit(1)
            self.frame_height, self.frame_width = frame.shape[:2]
            self._initialize_video_writer_instance(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("No frame received. Exiting...")
                    break

                self.frame_number += 1
                self.frame: np.ndarray = frame.copy()  # Make a copy to draw annotations
                timestamp: datetime = datetime.now()
                self.timestamps.append(timestamp)

                # Perform object detection with increased confidence threshold for better accuracy
                results = self.model.predict(self.frame, conf=0.5, verbose=False)

                # Extract detected classes and count vehicles
                detections = results[0].boxes
                total_vehicles, vehicle_type_count = self._process_detections(detections)
                self.vehicle_counts.append(total_vehicles)
                self.vehicle_type_counts.append(vehicle_type_count)
                traffic_density = self.classify_traffic_density(total_vehicles)

                # Update traffic density counts for percentage calculation
                if traffic_density == 'High Traffic':
                    self.high_traffic_count += 1
                elif traffic_density == 'Medium Traffic':
                    self.medium_traffic_count += 1

                # Add legend panel
                self._add_legend_panel(traffic_density, total_vehicles, vehicle_type_count)

                # Write the annotated frame to the output video
                if self.video_writer is not None:
                    self.video_writer.write(self.frame)

                # Display the resulting frame with legend
                cv2.imshow('Real-Time Traffic Monitoring', self.frame)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exit signal received. Stopping video processing...")
                    break

                # Logging every 100 frames to avoid excessive log entries
                if self.frame_number % 100 == 0:
                    logging.info(f"Processed {self.frame_number} frames.")

            self._cleanup()

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Exiting gracefully...")
            self._cleanup()
        except Exception as e:
            logging.error(f"An error occurred during video processing: {e}")
            self._cleanup()
            sys.exit(1)

    def _initialize_video_writer_instance(self, frame: np.ndarray) -> None:
        """
        Initialize the VideoWriter with the correct frame size.

        Args:
            frame (np.ndarray): The first frame of the video.
        """
        try:
            fourcc: int = cv2.VideoWriter_fourcc(*'XVID')
            frame_height, frame_width, _ = frame.shape
            # Adjust output frame width to include legend
            output_frame_width: int = frame_width + int(frame_width * self.legend_ratio)
            self.video_writer = cv2.VideoWriter(
                str(self.output_video_path),
                fourcc,
                20.0,  # Assuming 20 FPS; adjust as needed
                (output_frame_width, frame_height)
            )
            if not self.video_writer.isOpened():
                logging.error(f"Failed to open VideoWriter with path: {self.output_video_path}")
                self.video_writer = None
            else:
                logging.info(f"VideoWriter initialized and saving to: {self.output_video_path.resolve()}")
        except Exception as e:
            logging.error(f"Failed to initialize VideoWriter instance: {e}")
            self.video_writer = None

    def _cleanup(self) -> None:
        """
        Release video capture, video writer, and destroy all OpenCV windows. Generate the report.
        """
        if self.lock:
            return  # Prevent re-entrance
        self.lock = True
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                logging.info("Video capture released.")
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
                logging.info("Video writer released.")
            cv2.destroyAllWindows()
            logging.info("All OpenCV windows destroyed.")

            if self.report_generator:
                # Generate the HTML report in a separate thread to avoid blocking
                report_thread: Thread = Thread(
                    target=self.report_generator.generate_html_report,
                    args=(self.timestamps, self.vehicle_counts, self.vehicle_type_counts),
                    daemon=True
                )
                report_thread.start()
                logging.info("Report generation started in a separate thread.")
        except Exception as e:
            logging.error(f"Failed during cleanup: {e}")
        finally:
            self.lock = False


def get_video_source() -> Union[str, int]:
    """
    Prompt the user to choose between uploading a video or using the webcam.

    Returns:
        Union[str, int]: Video source (file path or webcam index).
    """
    while True:
        print("\nSelect Video Source:")
        print("1. Upload a video file")
        print("2. Use webcam for real-time recording")
        choice: str = input("Enter your choice (1 or 2): ").strip()

        if choice == '1':
            file_path: str = input("Enter the path to the video file: ").strip()
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
        video_source: Union[str, int] = get_video_source()

        # Initialize and start the traffic monitor
        traffic_monitor: TrafficMonitor = TrafficMonitor(source=video_source)
        traffic_monitor.process_stream()

    except Exception as e:
        logging.error(f"An unexpected error occurred in the main function: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()