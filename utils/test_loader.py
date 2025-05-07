import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.video_loader import load_video_and_select_frame


from utils.video_loader import load_video_and_select_frame

frame, init_box, cap = load_video_and_select_frame("videos/match1.mp4")
print(f"Initial box: {init_box}")
