import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

def load_model():
    model = YOLO("best.pt")
    return model

def process_frame(frame, model):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame

def process_video_file(uploaded_file, model):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(frame, model)

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        stframe.image(frame_pil, caption="Live Fall Detection", use_container_width=True)

    cap.release()

def process_live_camera(model):
    
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(frame, model)

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        stframe.image(frame_pil, caption="Live Fall Detection", use_container_width=True)

    cap.release()

def main():
    st.title("Fall Detection - YOLOv8 Live Prediction")
    
    st.sidebar.header("Choose Input Source")
    input_source = st.sidebar.radio("Select input source", ["Upload Video", "Live Camera"])

    # Load YOLOv8 model
    model = load_model()

    if input_source == "Upload Video":
        st.sidebar.header("Upload Video")
        uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            process_video_file(uploaded_file, model)
            st.success("Video processing completed!")
        else:
            st.warning("Please upload a video to start predictions.")

    elif input_source == "Live Camera":
        st.sidebar.header("Live Camera Feed")
        st.warning("Make sure your camera is working.")
        process_live_camera(model)

if __name__ == "__main__":
    main()





# import streamlit as st
# import cv2
# import torch
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import tempfile

# def load_model():
#     model = YOLO("best.pt")  # Replace 'best.pt' with your YOLOv8 model file path
#     return model

# def process_frame(frame, model):
#     results = model(frame)  # Run inference on the frame
#     annotated_frame = results[0].plot()  # Draw bounding boxes and labels
#     return annotated_frame

# def main():
#     st.title("Fall Detection - YOLOv8 Live Prediction")
    
#     st.sidebar.header("Upload Video")
#     uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
#     if uploaded_file is not None:
#         # Save uploaded file temporarily
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())

#         # Load YOLOv8 model
#         model = load_model()

#         # Open video
#         cap = cv2.VideoCapture(tfile.name)

#         stframe = st.empty()  # Placeholder for displaying frames

#         # Process video frame-by-frame
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break  # End of video

#             # Perform prediction
#             annotated_frame = process_frame(frame, model)

#             # Convert the annotated frame for Streamlit display
#             frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#             frame_pil = Image.fromarray(frame_rgb)
#             stframe.image(frame_pil, caption="Live Fall Detection", use_container_width=True)

#         cap.release()
#         st.success("Video processing completed!")
#     else:
#         st.warning("Please upload a video to start predictions.")

# if __name__ == "__main__":
#     main()
