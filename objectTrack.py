import cv2
import threading
import time
from collections import deque

# Global variables
bboxes = []
bbox_names = []
current_bbox = [0, 0, 0, 0]
selecting = False
start_tracking = True  # Default to instant mode
instant_mode = True  # Default to instant mode
feature_mode = False
trackers = []
tracking_frame = None
display_frame = None
fps = 0
capture_thread_running = True
tracking_data = []
tracking_updated = threading.Condition()
tracking_queue = deque(maxlen=1)  # Queue to hold the latest tracking frames
features = []  # Store initial features of tracked objects

# Function to initialize the tracker
def create_tracker(tracker_type="CSRT"):
    tracker_types = {
        "CSRT": cv2.legacy.TrackerCSRT_create,
        "KCF": cv2.legacy.TrackerKCF_create,
        "MIL": cv2.legacy.TrackerMIL_create,
        "TLD": cv2.legacy.TrackerTLD_create,
        "MedianFlow": cv2.legacy.TrackerMedianFlow_create,
        "MOSSE": cv2.legacy.TrackerMOSSE_create
    }
    if tracker_type in tracker_types:
        return tracker_types[tracker_type]()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

# Mouse callback function to get the initial bounding box
def select_points(event, x, y, flags, param):
    global current_bbox, selecting, start_tracking, instant_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        current_bbox = [x, y, x, y]
        if not instant_mode:
            start_tracking = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            current_bbox[2] = x
            current_bbox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        current_bbox[2] = x
        current_bbox[3] = y
        bboxes.append(tuple(current_bbox))
        if instant_mode:
            bbox_names.append(f"Object {len(bbox_names) + 1}")
            add_tracker(current_bbox)
            capture_features(current_bbox)  # Capture initial features
            current_bbox = [0, 0, 0, 0]  # Clear current bounding box

def add_tracker(bbox):
    global trackers, tracking_frame, features
    if tracking_frame is not None:
        (x0, y0, x1, y1) = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(display_frame.shape[1], x1), min(display_frame.shape[0], y1)
        tracker = create_tracker("CSRT")
        trackers.append(tracker)
        tracker.init(tracking_frame, (x0, y0, x1 - x0, y1 - y0))
        features.append(tracking_frame[y0:y1, x0:x1])

def capture_features(bbox):
    global features, tracking_frame
    (x0, y0, x1, y1) = bbox
    roi = tracking_frame[y0:y1, x0:x1]
    features.append(roi)

# Function to capture frames for display and tracking
def capture_frames():
    global display_frame, tracking_frame, capture_thread_running, tracking_updated
    cap = cv2.VideoCapture(2) # Change cameras here, 1 will be your default camera, anything after 1 will be dependant on other factors, try em all
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while capture_thread_running:
        ret, frame = cap.read()
        if not ret:
            break
        flipped_frame = cv2.flip(frame, 1)
        with tracking_updated:
            display_frame = flipped_frame.copy()
            tracking_frame = flipped_frame.copy()
            tracking_updated.notify_all()
    cap.release()

# Function to process the frames and track the object
def process_frames():
    global bboxes, selecting, start_tracking, trackers, tracking_frame, tracking_data, fps, capture_thread_running, tracking_updated, tracking_queue, features

    prev_time = time.time()

    while capture_thread_running:
        with tracking_updated:
            tracking_updated.wait()

        if tracking_frame is None:
            continue

        if start_tracking or instant_mode:
            tracking_data = []
            for i, tracker in enumerate(trackers):
                success, box = tracker.update(tracking_frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    tracking_data.append((x, y, w, h, True))
                else:
                    # Attempt to reacquire object
                    (x, y, w, h) = reacquire_object(i)
                    if (x, y, w, h) != (0, 0, 0, 0):
                        tracker.init(tracking_frame, (x, y, w, h))
                        tracking_data.append((x, y, w, h, True))
                    else:
                        tracking_data.append((0, 0, 0, 0, False))

            # Update the tracking queue
            tracking_queue.append(tracking_data)

        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time > 0:
            fps = 1 / elapsed_time
        prev_time = current_time

        # Run feature comparison mode
        if feature_mode:
            for i in range(len(trackers)):
                (x, y, w, h) = reacquire_object(i)
                if (x, y, w, h) != (0, 0, 0, 0):
                    trackers[i].init(tracking_frame, (x, y, w, h))

def reacquire_object(index):
    global features, tracking_frame
    if index >= len(features):
        return (0, 0, 0, 0)
    
    feature = features[index]
    res = cv2.matchTemplate(tracking_frame, feature, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.6:  # Threshold to determine if the object is found
        (x, y) = max_loc
        (h, w, _) = feature.shape
        return (x, y, w, h)
    return (0, 0, 0, 0)

def main():
    global capture_thread_running, display_frame, tracking_data, current_bbox, bboxes, bbox_names, selecting, start_tracking, trackers, tracking_queue, instant_mode, feature_mode

    # Start the frame capture thread
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.start()

    # Start the frame processing thread
    process_thread = threading.Thread(target=process_frames)
    process_thread.start()

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", select_points)

    blink_counter = 0
    previous_tracking_data = []

    print("Press 'i' to enable named tracking mode. In named tracking mode, objects are tracked with names.")
    print("Press 'f' to toggle feature comparison mode. In this mode, objects will be reacquired using feature matching if tracking is lost.")

    while True:
        if display_frame is not None:
            display_frame_copy = display_frame.copy()

            # Draw current selection box if selecting
            if selecting:
                (x0, y0, x1, y1) = current_bbox
                cv2.rectangle(display_frame_copy, (x0, y0), (x1, y1), (255, 0, 0), 3)
                if not instant_mode:
                    cv2.putText(display_frame_copy, "Press 's' to start tracking", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Draw all selected boxes if not in instant mode
            if not instant_mode:
                for box in bboxes:
                    (x0, y0, x1, y1) = box
                    cv2.rectangle(display_frame_copy, (x0, y0), (x1, y1), (255, 0, 0), 3)

            # Draw tracking data
            if start_tracking or instant_mode:
                num_objects = 0
                if tracking_queue:
                    previous_tracking_data = tracking_queue[-1]  # Get the latest tracking data
                for i, data in enumerate(previous_tracking_data):
                    x, y, w, h, success = data
                    if success:
                        num_objects += 1
                        cv2.rectangle(display_frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        if i < len(bbox_names):
                            name = bbox_names[i]
                            cv2.putText(display_frame_copy, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    else:
                        if i < len(bbox_names):
                            name = bbox_names[i]
                            # Blinking "Searching for" text
                            if blink_counter % 20 < 10:
                                cv2.putText(display_frame_copy, f"Searching for {name}", (10, 140 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                if num_objects > 0:
                    # Blinking "TRACKING ACTIVE" text
                    blink_counter += 1
                    if blink_counter % 20 < 10:
                        cv2.putText(display_frame_copy, "TRACKING ACTIVE", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # Display stats
                cv2.putText(display_frame_copy, f"Objects: {num_objects}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                for i, data in enumerate(previous_tracking_data):
                    x, y, w, h, success = data
                    if success:
                        if i < len(bbox_names):
                            name = bbox_names[i]
                            cv2.putText(display_frame_copy, f"{name}: (x={x}, y={y}, w={w}, h={h})", (10, 330 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame_copy, "Awaiting box to be drawn", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Display FPS
            cv2.putText(display_frame_copy, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Display Instant Mode Status
            cv2.putText(display_frame_copy, f"Instant Mode: {'ON' if instant_mode else 'OFF'}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(display_frame_copy, "Press 'i' to enable named tracking mode", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Display Feature Mode Status
            cv2.putText(display_frame_copy, f"Feature Mode: {'ON' if feature_mode else 'OFF'}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(display_frame_copy, "Press 'f' to enable feature mode", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Display Key Toggle Descriptions
            cv2.putText(display_frame_copy, "Press 'q' to quit", (10, display_frame_copy.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(display_frame_copy, "Press 'ESC' to reset", (10, display_frame_copy.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(display_frame_copy, "Press 's' to start tracking", (10, display_frame_copy.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Frame", display_frame_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                capture_thread_running = False
                break
            elif key == ord('s') and not instant_mode and bboxes:
                for box in bboxes:
                    name = input(f"Enter name for the object at position {box}: ")
                    bbox_names.append(name)
                    add_tracker(box)
                    capture_features(box)  # Capture initial features
                start_tracking = True
                bboxes.clear()
            elif key == ord('i'):
                instant_mode = not instant_mode
                start_tracking = not instant_mode  # Adjust start_tracking based on mode
                print(f"Named tracking mode {'enabled' if not instant_mode else 'disabled'}")
            elif key == ord('f'):
                feature_mode = not feature_mode
                print(f"Feature mode {'enabled' if feature_mode else 'disabled'}")
            elif key == 27:  # ESC key
                start_tracking = False
                bboxes.clear()
                bbox_names.clear()  # Clear names when resetting tracking
                trackers.clear()  # Clear trackers when resetting tracking
                features.clear()  # Clear features when resetting tracking

    capture_thread_running = False
    capture_thread.join()
    process_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
