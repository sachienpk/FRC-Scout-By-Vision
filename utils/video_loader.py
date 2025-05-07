import cv2

def load_video_and_select_frame(video_path):
    """
    Loads a video, pauses on a frame, and lets the user click on the robot to track.
    
    Returns:
        - frame: the selected video frame (image)
        - init_box: (x, y, w, h) for the initial tracking rectangle
        - cap: the video capture object (to continue reading frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Go to frame 30 (optional — can be adjusted)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to read frame from video.")

    clicked = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            cv2.destroyAllWindows()

    cv2.imshow("Click on the robot to track", frame)
    cv2.setMouseCallback("Click on the robot to track", click_event)
    cv2.waitKey(0)

    if not clicked:
        raise Exception("No click detected.")

    x, y = clicked[0]
    box_size = 40  # pixels
    init_box = (x - box_size // 2, y - box_size // 2, box_size, box_size)

    return frame, init_box, cap
