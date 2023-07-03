# Visionoscope Workbench

import time

# Built-in packages
import PIL
# External packages
import cv2
import streamlit as st
from ultralytics import YOLO

# Local modules
import settings

TITLE = 'Visionoscope Workbench'

# Models
DETECT_MODEL = 'Object Detection'
SEGMENT_MODEL = 'Object Segmentation'
POSE_MODEL = 'Pose Detection'
MODEL_LIST = [DETECT_MODEL, SEGMENT_MODEL, POSE_MODEL]

# Weights
NANO_WEIGHT = 'Nano'
SMALL_WEIGHT = 'Small'
MEDIUM_WEIGHT = 'Medium'
LARGE_WEIGHT = 'Large'
EXTRA_LARGE_WEIGHT = 'Extra Large'
WEIGHT_LIST = [NANO_WEIGHT, SMALL_WEIGHT, MEDIUM_WEIGHT, LARGE_WEIGHT, EXTRA_LARGE_WEIGHT]

# Input sources
IMAGE_SOURCE = 'Image'
VIDEO_SOURCE = 'Video'
WEBCAM_SOURCE = 'Webcam'
RTSP_SOURCE = 'RTSP'
SOURCE_LIST = [IMAGE_SOURCE, VIDEO_SOURCE, WEBCAM_SOURCE, RTSP_SOURCE]

# Supported image file extensions
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'webp']

# Page Configuration
st.set_page_config(page_title=TITLE, page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# Title
st.title(TITLE)

# Sidebar 1st header
st.sidebar.header("Model Settings")

# Model selection
model_type = st.sidebar.radio("Select Model", MODEL_LIST)

# Model weight selection
model_weight = st.sidebar.radio("Select Model Weight", WEIGHT_LIST)

# Model Confidence selection
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# Tracker selection
tracker_type = st.sidebar.radio("Select Tracker", ["bytetrack.yaml", "botsort.yaml", "No"])

# Setting model suffix
if model_type == DETECT_MODEL:
    MODEL_SUFFIX = ''
elif model_type == SEGMENT_MODEL:
    MODEL_SUFFIX = '-seg'
elif model_type == POSE_MODEL:
    MODEL_SUFFIX = '-pose'
else:
    st.error('Failed to select model!')

# Setting model weight suffix
if model_weight == NANO_WEIGHT:
    MODEL_WEIGHT_SUFFIX = 'n'
elif model_weight == SMALL_WEIGHT:
    MODEL_WEIGHT_SUFFIX = 's'
elif model_weight == MEDIUM_WEIGHT:
    MODEL_WEIGHT_SUFFIX = 'm'
elif model_weight == LARGE_WEIGHT:
    MODEL_WEIGHT_SUFFIX = 'l'
elif model_weight == EXTRA_LARGE_WEIGHT:
    MODEL_WEIGHT_SUFFIX = 'x'
else:
    st.error('Failed to select model weight!')

# Construct model path
model_path = str(settings.MODEL_DIRECTORY) + '/yolov8' + MODEL_WEIGHT_SUFFIX + MODEL_SUFFIX + '.pt'

# Loading Pre-trained Model
try:
    model = YOLO(model_path)
except Exception as exception:
    st.error(f"Unable to load model. Check the specified path: {model_path}: {exception}")

# Sidebar 2nd header
st.sidebar.header("Input Settings")

# Input source selection
source_radio = st.sidebar.radio("Select Source", SOURCE_LIST)


def display_result_frames(st_frame, image):
    # Display object tracking, if specified
    if tracker_type == 'No':
        resource = model.predict(image, conf=confidence, persist=True)
    else:
        resource = model.track(image, conf=confidence, persist=True, tracker=tracker_type)

    # Plot the result objects on the video frame
    plotted_resource = resource[0].plot()
    st_frame.image(plotted_resource, caption='Result Video', channels="BGR", use_column_width=True)


source_image = None

# If image is selected
if source_radio == IMAGE_SOURCE:
    source_image = st.sidebar.file_uploader("Select an Image File", type=IMAGE_EXTENSIONS, accept_multiple_files=False)

    column_1, column_2 = st.columns(2)

    with column_1:
        try:
            if source_image is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_image)
                st.image(source_image, caption="Uploaded Image", use_column_width=True)
        except Exception as exception:
            st.error(f"Error occurred while opening the image: {exception}")
    with column_2:
        if source_image is None:
            default_result_image_path = str(settings.DEFAULT_RESULT_IMAGE)
            default_result_image = PIL.Image.open(
                default_result_image_path)
            st.image(default_result_image_path, caption='Result Image', use_column_width=True)
        else:
            if st.sidebar.button('Run'):
                if tracker_type == 'No':
                    resource = model.predict(uploaded_image, conf=confidence, persist=True)
                else:
                    resource = model.track(uploaded_image, conf=confidence, persist=True, tracker=tracker_type)
                boxes = resource[0].boxes
                plotted_resource = resource[0].plot()[:, :, ::-1]
                st.image(plotted_resource, caption='Result Image', use_column_width=True)
                st.snow()
                try:
                    with st.expander("Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as exception:
                    st.write("No image is uploaded yet!")
                    st.write(exception)
elif source_radio == VIDEO_SOURCE:
    source_video = st.sidebar.selectbox("Select a Video File", settings.VIDEO_LIST.keys())

    column_1, column_2 = st.columns(2)

    with column_1:
        with open(settings.VIDEO_LIST.get(source_video), 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
    with column_2:
        if st.sidebar.button('Run'):
            try:
                video_capture = cv2.VideoCapture(
                    str(settings.VIDEO_LIST.get(source_video)))
                st_frame = st.empty()
                while video_capture.isOpened():
                    success, image = video_capture.read()
                    if success:
                        display_result_frames(st_frame, image)
                    else:
                        video_capture.release()
                        break
            except Exception as exception:
                st.error(f"Error loading video: {exception}")
elif source_radio == WEBCAM_SOURCE:
    source_webcam = st.sidebar.number_input("Webcam Serial", format="%d", min_value=0, value=0, step=1)
    if st.sidebar.button('Run'):
        try:
            video_capture = cv2.VideoCapture(int(source_webcam))
            st_frame = st.empty()
            while video_capture.isOpened():
                success, image = video_capture.read()
                if success:
                    display_result_frames(st_frame, image)
                else:
                    video_capture.release()
                    break
        except Exception as exception:
            st.error(f"Error loading video: {exception}")
elif source_radio == RTSP_SOURCE:
    source_rtsp = st.sidebar.text_input("RTSP Stream URL")
    if st.sidebar.button('Run'):
        try:
            video_capture = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while video_capture.isOpened():
                success, image = video_capture.read()
                if success:
                    display_result_frames(st_frame, image)
                else:
                    video_capture.release()
                    break
        except Exception as exception:
            st.error(f"Error loading RTSP stream: {exception}")

else:
    st.error("Please select a valid source type!")
