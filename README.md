# objectTrack
ObjectTrack is a Python script using OpenCV (cv2) to track any object within a video frame. Once a bounding box is drawn around the object, it tracks it until it goes off-screen. The script then searches for the original captured features to reidentify and resume tracking the object when it returns to view.
