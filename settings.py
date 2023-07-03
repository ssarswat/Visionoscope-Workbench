# Visionoscope Workbench

# Built-in packages
import sys
from pathlib import Path

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Model directory path
MODEL_DIRECTORY = ROOT / 'YOLO_weights'

# Default image files' path
IMAGES_DIRECTORY = ROOT / 'demo_files/images'
DEFAULT_IMAGE = IMAGES_DIRECTORY / 'default.jpg'
DEFAULT_RESULT_IMAGE = IMAGES_DIRECTORY / 'default_result.jpg'

# Default video files' path
VIDEO_DIRECTORY = ROOT / 'demo_files/videos'
VIDEO_1_PATH = VIDEO_DIRECTORY / 'video_1.mp4'
VIDEO_2_PATH = VIDEO_DIRECTORY / 'video_2.mp4'
VIDEO_LIST = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH
}
