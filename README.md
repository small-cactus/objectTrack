# <img src="objectTrack.png" width="40" alt="objectTrack" align="top" /> objectTrack
> This Python script provides advanced object tracking using OpenCV. It supports multiple tracking algorithms, dynamic object reacquisition, and real-time feature comparison. The system allows users to select objects to track with a mouse, supports named tracking, and includes multiple modes for enhanced performance and reliability.

## Features
- **Instant Mode**: Track objects immediately upon selection with a mouse.
- **Named Tracking**: Assign names to tracked objects for easier identification.
- **Feature Comparison Mode**: Reacquire lost objects using initial features captured during selection.
- **Real-Time Multithreaded Processing**: Capture and process frames in real-time for minimal latency.
- **Flexible Tracking Algorithms**: Choose from multiple tracking algorithms like CSRT, KCF, MIL, and more.

## Installation

### Prerequisites
Ensure you have Python 3.6 or newer installed. The script has been tested with Python 3.11.

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/small-cactus/objectTrack.git
   cd objectTrack
   ```

2. **Set Up a Python Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate   # On macOS and Linux
   ```

3. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   ```

   Libraries to be installed:
   ```
   opencv-contrib-python==4.10.0.82
   ```

## Usage
Run the script from your command line:
```bash
python3 objectTracking.py
```

### Key Bindings
- **'i'**: Toggle named tracking mode.
- **'f'**: Toggle feature comparison mode.
- **'s'**: Start tracking (in non-instant mode).
- **'ESC'**: Reset all tracking.
- **'q'**: Quit the application.

### Configuration
Adjust parameters based on your setup:
- **Camera Index**: Change `cap = cv2.VideoCapture(0)` if the default camera is not used. `0` is the value you should be changing.
- **Resolution**: Set the desired resolution with `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)` and `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
