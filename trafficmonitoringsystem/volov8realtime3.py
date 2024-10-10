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
import tkinter as tk
from PIL import Image, ImageTk
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants for traffic density thresholds
LOW_THRESHOLD: int = 5      # Less than 10 vehicles per frame
MEDIUM_THRESHOLD: int = 10   # 10 to 30 vehicles per frame

# Vehicle classes based on YOLOv8's COCO dataset
VEHICLE_CLASSES: set = {'car', 'truck', 'bus', 'motorbike', 'bicycle'}

# Detection confidence threshold
DETECTION_CONFIDENCE: float = 0.5

# Distance threshold for tracking
TRACKING_DISTANCE: float = 10.0

# Maximum frames to keep an object without updates
MAX_MISSING_FRAMES: int = 10


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
        model_path: str = 'yolov8n.pt',
        output_folder: str = 'traffic_reports'
    ) -> None:
        self.source: Union[str, int] = source
        self.output_folder: Path = Path(output_folder)
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
        self.legend_ratio: float = 0.1  # 10% of the frame width for legend
        self.colors: Dict[str, Tuple[int, int, int]] = {
            'Low Traffic': (0, 255, 0),
            'Medium Traffic': (0, 165, 255),
            'High Traffic': (0, 0, 255)
        }
        self.tracked_objects: Dict[int, Dict[str, Union[str, Tuple[int, int, int, int], int]]] = {}
        self.next_object_id: int = 0  # ID to assign to the next detected object
        self.lock: bool = False  # Simple lock to prevent concurrent access during cleanup
        self.missing_frames_counter: Dict[int, int] = {}
        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize the traffic monitor by setting up the output folder and loading the YOLO model.
        """
        self._create_output_folder()
        self._load_yolo_model()
        self.report_generator = ReportGenerator(self.output_folder)

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
        current_frame_ids: List[int] = []

        for det in detections:
            try:
                cls_id: int = int(det.cls[0])
                cls_name: str = self.model.names.get(cls_id, '')
                if cls_name in VEHICLE_CLASSES and det.conf[0] >= DETECTION_CONFIDENCE:
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
                    object_id = self._assign_object_id(cls_name, (x1, y1, x2, y2))
                    if object_id is not None:
                        current_frame_ids.append(object_id)
            except Exception as e:
                logging.error(f"Error processing detection: {e}")

        # Update tracked objects, remove those missing for too long
        self._update_tracked_objects(current_frame_ids)

        unique_vehicles = len(self.tracked_objects)
        return unique_vehicles, dict(vehicle_type_count)

    def _assign_object_id(self, cls_name: str, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """
        Assign a unique ID to the detected vehicle and update its position.

        Args:
            cls_name (str): Class name of the detected vehicle.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates of the detected vehicle.

        Returns:
            Optional[int]: Assigned object ID or None if tracking fails.
        """
        x1, y1, x2, y2 = bbox
        bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        assigned_id: Optional[int] = None
        min_distance: float = float('inf')

        # Find the closest existing object
        for obj_id, obj_info in self.tracked_objects.items():
            distance = np.linalg.norm(bbox_center - np.array(obj_info['center']))
            if distance < TRACKING_DISTANCE and distance < min_distance:
                min_distance = distance
                assigned_id = obj_id

        if assigned_id is not None:
            # Update the existing object's position
            self.tracked_objects[assigned_id]['bbox'] = bbox
            self.tracked_objects[assigned_id]['center'] = bbox_center.tolist()
            self.missing_frames_counter[assigned_id] = 0
        else:
            # Assign a new object ID
            self.tracked_objects[self.next_object_id] = {
                'class': cls_name,
                'bbox': bbox,
                'center': bbox_center.tolist()
            }
            self.missing_frames_counter[self.next_object_id] = 0
            assigned_id = self.next_object_id
            self.next_object_id += 1

        return assigned_id

    def _update_tracked_objects(self, current_frame_ids: List[int]) -> None:
        """
        Update the tracked objects by removing those that have not been detected for a specified number of frames.

        Args:
            current_frame_ids (List[int]): List of object IDs detected in the current frame.
        """
        to_remove: List[int] = []
        for obj_id in self.tracked_objects.keys():
            if obj_id not in current_frame_ids:
                self.missing_frames_counter[obj_id] += 1
                if self.missing_frames_counter[obj_id] > MAX_MISSING_FRAMES:
                    to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            del self.missing_frames_counter[obj_id]

    def _add_legend_panel(
        self,
        traffic_density: str,
        total_vehicles: int,
        vehicle_type_counts: Dict[str, int]
    ) -> None:
        """
        Adds a legend panel on the right side of the frame displaying vehicle counts and traffic density.

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

            # Display vehicle type counts
            y_offset = 150
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

    def _draw_detection_zones(self) -> None:
        """
        Draws detection zones on the frame for each highway lane to ensure balanced detection.
        """
        try:
            zone_thickness = 2
            zone_color = (255, 0, 0)
            # Assuming double highway: two zones side by side
            zone_width = self.frame_width // 2
            cv2.line(self.frame, (zone_width, 0), (zone_width, self.frame_height), zone_color, zone_thickness)
            cv2.putText(
                self.frame, "Lane 1", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 2
            )
            cv2.putText(
                self.frame, "Lane 2", (zone_width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 2
            )
        except Exception as e:
            logging.error(f"Failed to draw detection zones: {e}")

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

            # Retrieve frame dimensions
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to read from video source.")
                sys.exit(1)
            self.frame_height, self.frame_width = frame.shape[:2]
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

                # Draw detection zones
                self._draw_detection_zones()

                # Perform object detection with increased confidence threshold for better accuracy
                results = self.model.predict(self.frame, conf=DETECTION_CONFIDENCE, verbose=False)
                
                # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
                detections = self._apply_nms(results[0].boxes)

                # Extract detected classes and count vehicles
                total_vehicles, vehicle_type_count = self._process_detections(detections)
                self.vehicle_counts.append(total_vehicles)
                self.vehicle_type_counts.append(vehicle_type_count)
                traffic_density = self.classify_traffic_density(total_vehicles)

                # Add legend panel
                self._add_legend_panel(traffic_density, total_vehicles, vehicle_type_count)

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

    def _apply_nms(self, boxes) -> List:
        """
        Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        Args:
            boxes: Detected bounding boxes.

        Returns:
            List of filtered detections after NMS.
        """
        try:
            # Extract bounding boxes, confidences, and class IDs
            bbox = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            indices = cv2.dnn.NMSBoxes(
                bboxes=bbox.tolist(),
                scores=scores.tolist(),
                score_threshold=DETECTION_CONFIDENCE,
                nms_threshold=0.4
            )

            filtered_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    det = boxes[i]
                    filtered_detections.append(det)

            return filtered_detections
        except Exception as e:
            logging.error(f"Failed to apply NMS: {e}")
            return []

    def _cleanup(self) -> None:
        """
        Release video capture and destroy all OpenCV windows. Generate the report.
        """
        if self.lock:
            return  # Prevent re-entrance
        self.lock = True
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                logging.info("Video capture released.")
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


class TrafficMonitorGUI(TrafficMonitor):
    def __init__(
        self,
        source: Union[str, int],
        model_path: str = 'yolov8n.pt',
        output_folder: str = 'traffic_reports'
    ) -> None:
        super().__init__(source, model_path, output_folder)
        self.root = tk.Tk()
        self.root.title("Real-Time Traffic Monitoring")
        self.setup_gui()
        self.video_thread = Thread(target=self.process_stream, daemon=True)
        self.video_thread.start()
        self.update_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def setup_gui(self) -> None:
        """
        Sets up the GUI layout with video and statistics panels.
        """
        # Video Display Panel
        self.video_panel = tk.Label(self.root)
        self.video_panel.pack(side="left", padx=10, pady=10)

        # Statistics Panel
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Total Vehicles
        self.total_label = tk.Label(self.stats_frame, text="Total Vehicles: 0", font=("Helvetica", 16))
        self.total_label.pack(pady=10)

        # Traffic Density
        self.density_label = tk.Label(self.stats_frame, text="Traffic Density: Low", font=("Helvetica", 16))
        self.density_label.pack(pady=10)

        # Vehicle Types
        self.type_labels = {}
        for vehicle_type in sorted(VEHICLE_CLASSES):
            label = tk.Label(
                self.stats_frame, text=f"{vehicle_type.capitalize()}: 0", font=("Helvetica", 14)
            )
            label.pack(pady=5)
            self.type_labels[vehicle_type] = label

    def update_gui(self) -> None:
        """
        Periodically updates the GUI with the latest frame and statistics.
        """
        if hasattr(self, 'frame') and self.frame is not None:
            # Convert the OpenCV frame to PIL Image
            cv2_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_image)

            # Update the video panel
            self.video_panel.imgtk = imgtk
            self.video_panel.configure(image=imgtk)

        # Update statistics if available
        if self.vehicle_counts and self.vehicle_type_counts:
            latest_total = self.vehicle_counts[-1]
            latest_density = self.classify_traffic_density(latest_total)
            latest_types = self.vehicle_type_counts[-1]

            self.total_label.config(text=f"Total Vehicles: {latest_total}")
            self.density_label.config(text=f"Traffic Density: {latest_density}")

            for vehicle_type, label in self.type_labels.items():
                count = latest_types.get(vehicle_type, 0)
                label.config(text=f"{vehicle_type.capitalize()}: {count}")

        # Schedule the next update
        self.root.after(30, self.update_gui)  # Update every 30 ms

    def process_stream(self) -> None:
        """
        Overrides the parent method to remove OpenCV window and use Tkinter instead.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logging.error(f"Cannot open video source: {self.source}")
                sys.exit(1)
            else:
                logging.info(f"Video source '{self.source}' opened successfully.")

            # Retrieve frame dimensions
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to read from video source.")
                sys.exit(1)
            self.frame_height, self.frame_width = frame.shape[:2]
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

                # Draw detection zones
                self._draw_detection_zones()

                # Perform object detection
                results = self.model.predict(self.frame, conf=DETECTION_CONFIDENCE, verbose=False)

                # Apply Non-Maximum Suppression (NMS)
                detections = self._apply_nms(results[0].boxes)

                # Extract detected classes and count vehicles
                total_vehicles, vehicle_type_count = self._process_detections(detections)
                self.vehicle_counts.append(total_vehicles)
                self.vehicle_type_counts.append(vehicle_type_count)
                traffic_density = self.classify_traffic_density(total_vehicles)

                # Add legend panel (optional in GUI)
                # self._add_legend_panel(traffic_density, total_vehicles, vehicle_type_count)

                # Optionally, add legend or other OpenCV annotations
                # For GUI integration, it's better to handle annotations via Tkinter

                # Logging every 100 frames to avoid excessive log entries
                if self.frame_number % 100 == 0:
                    logging.info(f"Processed {self.frame_number} frames.")

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Exiting gracefully...")
            self._cleanup()
        except Exception as e:
            logging.error(f"An error occurred during video processing: {e}")
            self._cleanup()
            sys.exit(1)

    def _cleanup(self) -> None:
        """
        Release video capture and perform any necessary cleanup.
        """
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                logging.info("Video capture released.")
            self.root.quit()
            logging.info("GUI closed.")
        except Exception as e:
            logging.error(f"Failed during cleanup: {e}")

    def on_close(self) -> None:
        """
        Handles the GUI window closing event.
        """
        self._cleanup()
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


# Modify your main function to use TrafficMonitorGUI
def main() -> None:
    """
    Main function to set up and run the traffic monitoring system based on user choice.
    """
    try:
        # Get user choice for video source
        video_source: Union[str, int] = get_video_source()

        # Initialize and start the traffic monitor GUI
        TrafficMonitorGUI(source=video_source)

    except Exception as e:
        logging.error(f"An unexpected error occurred in the main function: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()