import streamlit as st
import cv2
import os
import playsound
import torch
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
from twilio.rest import Client
import json
from deepface import DeepFace
import pages as pg
import time
from collections import deque
import threading


# Twilio configuration
#ACCOUNT_SID = 'AC68db1b58d30f94f25e922747a0689e0a'
#AUTH_TOKEN = '0394a2d70783b002e90f57d8444f030d'
#TWILIO_PHONE_NUMBER = '+19377212788'
#CARETAKER_NUMBER = '+919384166659'
# Please change the Twilio API credentials

# Parameters
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.75
ALERT_SOUND = "1secondalert.mp3"
PROFILE_FILE = "profiles.json"

# Define the admin credentials
USERNAME = "admin"
PASSWORD = "admin"

recent_alerts = deque(maxlen=5)

# Load models
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")
 
@st.cache_resource
def load_fall_classifier():
    classifier = load_model("fall_model.h5")
    clf = joblib.load("fall_detection_model.pkl")
    return classifier, clf

# Load profile data
def load_profiles():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            return json.load(file)
    return []

def get_latest_profile():
    profiles = load_profiles()
    return profiles[-1] if profiles else None

def get_registered_details():
    profiles = load_profiles()
    if profiles:
        return profiles[-1]
    return None

def play_alert():
    try:
        playsound.playsound(ALERT_SOUND)
    except Exception as e:
        print(f"Error playing sound: {e}")
        
# Save user profile
def save_profile(name, relative_name, age, phone, image):
    profile_data = {"name": name, "relative_name": relative_name, "age": age, "phone": phone, "image": image}
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
    else:
        profiles = []
    profiles.append(profile_data)
    with open(PROFILE_FILE, "w") as file:
        json.dump(profiles, file, indent=4)

# Load registered phone number
def get_registered_phone():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
        return profiles[-1]["phone"] if profiles else None
    return None

def get_registered_relative_name():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
        return profiles[-1]["relative_name"] if profiles else None
    return None

# Get the registered name
def get_registered_name():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
        return profiles[-1]["name"] if profiles else None
    return None

def get_registered_age():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
        return profiles[-1]["age"] if profiles else None
    return None

# Send alert
def send_alert(person_name=None, relative_name=None, phone_number=None):

    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    if not person_name or not relative_name or not phone_number:
        message = client.messages.create(
            body=f"Fall Detected! Please take immediate action!",
            from_=TWILIO_PHONE_NUMBER,
            to=CARETAKER_NUMBER
        )
        return
    if not phone_number.startswith('+'):
        phone_number = f"+91{phone_number}"
    
    message = client.messages.create(
        body=f"{person_name} has Fallen! Please take immediate action.",
        from_=TWILIO_PHONE_NUMBER,
        to=phone_number
    )
    st.warning(f"Fall Detected for Person {person_name}. Alert sent to {relative_name}!")

# Detect falls
FALL_ALERT_SENT = False  # Variable to track if the alert has already been sent

def recognize_face(frame):
    profiles = load_profiles()
    if not profiles:
        st.write("No profiles found for face recognition.")
        return None
    try:
        for profile in profiles:
            registered_face = profile["image"]
            result = DeepFace.verify(frame, registered_face, model_name='Facenet', enforce_detection=False)
            if result["distance"] < 0.3:
                st.write(f"Face verified successfully for {profile['name']}!")
                return profile  # Return the matched profile
    except Exception as e:
        st.write(f"Face recognition error: {e}")
    return None

frame_counter = 0
VERIFICATION_INTERVAL = 50 

def detect_fall(frame, model):
    global FALL_ALERT_SENT
    results = model(frame)
    fall_detected = False

    # Initialize face_verified in session state if not already set
    if 'face_verified' not in st.session_state:
        st.session_state.face_verified = None

    # Perform face verification only if no face has been verified yet
    # if st.session_state.face_verified is None:
    #     matched_profile = recognize_face(frame)
    #     if matched_profile:
    #         st.session_state.face_verified = matched_profile
    #         st.write(f"Face Verified: {matched_profile['name']}")

    if "frame_counter" not in st.session_state:
        st.session_state.frame_counter = 0

    if st.session_state.face_verified is None:
        if st.session_state.frame_counter % VERIFICATION_INTERVAL == 0:
            matched_profile = recognize_face(frame)
            if matched_profile:
                st.session_state.face_verified = matched_profile

        # Increment frame_counter in session state
        st.session_state.frame_counter += 1

    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            label = result.names[int(box.cls[0])]
            
            if confidence > CONFIDENCE_THRESHOLD and "Fall" in label:

                x_center, y_center, width, height = box.xywh[0]
                x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
                x2, y2 = int(x_center + width / 2), int(y_center + height / 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if not FALL_ALERT_SENT:
                    play_alert()
                    if st.session_state.face_verified:
                        st.write("Face Verified!!!")
                        person_name = st.session_state.face_verified["name"]
                        relative_name = st.session_state.face_verified["relative_name"]
                        phone_number = st.session_state.face_verified["phone"]
                        send_alert(person_name, relative_name, phone_number)
                        alert_message = f"🚨 {person_name} has fallen!"

                    else:
                        st.write("Face Not Verified !!!")
                        send_alert()
                        alert_message = "🚨 Unknown person has fallen!"

                    FALL_ALERT_SENT = True
                    fall_detected = True
                    recent_alerts.appendleft(alert_message)

    return fall_detected


def show_surveillance():

    st.markdown("""
    <h1 style='text-align: center; color: #2d3a54;'>DE-Fall</h1>
    <style>
        [data-testid = "stApp"]{
            background-image: url("https://img.freepik.com/free-vector/blue-pink-halftone-background_53876-99004.jpg?semt=ais_hybrid");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stImageCaption"]{
            color : #2d3a54;
            font-weight : bold !important ;
            font-size : 30px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    def stream_camera(rtsp_url, image_placeholder):
        cap = cv2.VideoCapture(rtsp_url)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(frame, use_container_width=True)
        cap.release()

    def display_camera(col, camera_name, rtsp_url, default_image):
        with col:
            st.markdown(f"**{camera_name}**")
            image_placeholder = st.empty()
            if st.button(f"Refresh {camera_name}"):
                stream_camera(rtsp_url, image_placeholder)
            else:
                image_placeholder.image(default_image, use_container_width=True)

    # Define the camera URLs
    camera_urls = {
        "Camera 1": 0,
        "Camera 2": "rtsp://admin:admin@192.168.176.126:1935",
        "Camera 3": "rtsp://staff:Sslab@123@172.17.137.103",
        "Camera 4": "rtsp://staff:Sslab@123@172.17.137.104",
        "Camera 5": "rtsp://staff:Sslab@123@172.17.137.105",
        "Camera 6": "rtsp://staff:Sslab@123@172.17.137.106",
    }

    default_image_path = "no_feed_available.jpg"
    default_image = Image.open(default_image_path)

    # Create a grid layout
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    display_camera(col1, "Camera 1", camera_urls["Camera 1"], default_image)
    display_camera(col2, "Camera 2", camera_urls["Camera 2"], default_image)
    display_camera(col3, "Camera 3", camera_urls["Camera 3"], default_image)
    display_camera(col4, "Camera 4", camera_urls["Camera 4"], default_image)
    display_camera(col5, "Camera 5", camera_urls["Camera 5"], default_image)
    display_camera(col6, "Camera 6", camera_urls["Camera 6"], default_image)




def show_falldetection():
        
        st.markdown("<h1 style='text-align: center; color: #2d3a54;'>DE-Fall</h1>", unsafe_allow_html=True)

        st.markdown(
            """
            <style>
                [data-testid = "stApp"]{
                    background-image: url("https://img.freepik.com/free-vector/blue-pink-halftone-background_53876-99004.jpg?semt=ais_hybrid");
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }

                [data-testid = "stHorizontalBlock"]{
                    height : 500px;
                }

                [class="stColumn st-emotion-cache-kvoai1 e6rk8up2"]{
                    background-color : #2d3a54;
                    padding : 20px;
                    border-radius : 5px;
                }

                [class="stColumn st-emotion-cache-7i1wy e6rk8up2"]{
                    background-color : #2d3a54;
                    padding : 20px;
                    border-radius : 5px;
                }

                [class="stHorizontalBlock st-emotion-cache-ocqkz7 e6rk8up0"]{
                    height : auto;
                }

                
            </style>
            """,
            unsafe_allow_html=True
        )

        # col1, col2 = st.columns([5, 1])

        # with col2 :
        #     st.subheader("Recent Alerts 🚨")
        #     alert_box = st.empty()  # Placeholder for updating alerts
            

        # with col1:

        #     st.header("Fall Detection System")
        #     model = load_yolo_model()
        #     option = st.radio("Choose an option:", ("Video Upload", "Live Input Feed"))

        #     if option == "Video Upload":
        #         if 'face_verified' in st.session_state:
        #             del st.session_state.face_verified  # Reset face verification
        #         uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        #         if uploaded_file:
        #             tfile = tempfile.NamedTemporaryFile(delete=False)
        #             tfile.write(uploaded_file.read())
        #             cap = cv2.VideoCapture(tfile.name)
        #             stframe = st.empty()

        #             while cap.isOpened():
        #                 ret, frame = cap.read()
        #                 if not ret:
        #                     break
        #                 detect_fall(frame, model)
        #                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #                 stframe.image(frame_rgb, caption="Live Fall Detection", use_container_width=True)
        #                 with col2:
        #                     alert_box.write("\n".join(recent_alerts) if recent_alerts else "No alerts")
        #             cap.release()
        #             st.success("Video processing completed!")

        #     elif option == "Live Input Feed":
        #         if 'face_verified' in st.session_state:
        #             del st.session_state.face_verified  # Reset face verification
        #         st.header("Live Fall Detection")
                
        #         start_button = st.button("Start Live Feed")
        #         stop_button = st.button("Stop Live Feed")
                
        #         if start_button:
        #             ip_camera_url = "rtsp://staff:Sslab@123@172.17.137.102"
        #             cap = cv2.VideoCapture(0)  # Open the webcam
        #             stframe = st.empty()
        #             while True:
        #                 ret, frame = cap.read()
        #                 if not ret or stop_button:
        #                     break
        #                 detect_fall(frame, model)
        #                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #                 stframe.image(frame_rgb, caption="Live Fall Detection", use_container_width=True)
        #                 with col2:
        #                     alert_box.write("\n".join(recent_alerts) if recent_alerts else "No alerts")
        #             cap.release()
        #             st.success("Live Feed Stopped!")
        
        camera_urls = {
            "Camera 1": 0,
            "Camera 2": "rtsp://admin:admin@192.168.176.126:1935",
            "Camera 3": 0, #"rtsp://staff:Sslab@123@172.17.137.103",
            "Camera 4": 0  #"rtsp://staff:Sslab@123@172.17.137.104"
        }

        # Default image when no feed is available
        default_image_path = "no_feed_available.jpg"  # Replace with the path to a suitable image

        col1, col2 = st.columns([4, 1])

        with col2:
            st.subheader("Recent Alerts 🚨")
            alert_box = st.empty()  # Placeholder for updating alerts

        with col1:
            st.header("Fall Detection System")
            model = load_yolo_model()
            option = st.radio("Choose an option:", ("Video Upload", "Live Input Feed"))

            if option == "Video Upload":
                if 'face_verified' in st.session_state:
                    del st.session_state.face_verified  # Reset face verification

                uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
                if uploaded_file:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_file.read())
                    cap = cv2.VideoCapture(tfile.name)
                    stframe = st.empty()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        detect_fall(frame, model)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        stframe.image(frame_rgb, caption="Live Fall Detection", use_container_width=True)
                        with col2:
                            alert_box.write("\n".join(recent_alerts) if recent_alerts else "No alerts")
                    cap.release()
                    st.success("Video processing completed!")

            elif option == "Live Input Feed":
                if 'face_verified' in st.session_state:
                    del st.session_state.face_verified  # Reset face verification
                st.header("Live Fall Detection")

                # Display camera selection buttons
                camera_selected = None
                col_buttons = st.columns(4)
                for i, cam in enumerate(camera_urls.keys()):
                    if col_buttons[i].button(cam, use_container_width=True):
                        camera_selected = cam

                stframe = st.empty()

                if camera_selected:
                    cap = cv2.VideoCapture(camera_urls[camera_selected])
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if cap.isOpened():
                        while True:
                            # Discard old frames to reduce lag
                            for _ in range(10):  # You can adjust this number as needed
                                cap.grab()

                            ret, frame = cap.read()
                            if not ret:
                                break

                            detect_fall(frame, model)

                            # Normalize and convert the frame for display
                            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            stframe.image(frame_rgb, caption=f"Live Feed: {camera_selected}", use_container_width=True)
                            
                            with col2:
                                alert_box.write("\n".join(recent_alerts) if recent_alerts else "No alerts")

                    else:
                        stframe.image(default_image_path, caption="No Feed Available", use_container_width=True)

                    cap.release()
                    st.success(f"Feed from {camera_selected} stopped.")


def show_register():
        
        st.markdown("<h1 style='text-align: center; color: white;'>DE-Fall</h1>", unsafe_allow_html=True)

        st.markdown(
            """
            <style>
                [data-testid = "stApp"]{
                    background-image: url("https://img.freepik.com/free-vector/blue-pink-halftone-background_53876-99004.jpg?semt=ais_hybrid");
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }

                [data-testid="stVerticalBlock"]{
                    background-color : #2d3a54;
                    padding : 20px;
                    border-radius : 5px;
                    width : 100%;
                }

                [data-testid="stTextInput"]{
                    width : 97% ;
                }

                [data-testid="stFileUploader"]{
                    width : 97% ;
                }

                [data-testid="stNumberInput"]{
                    width : 97% !important;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.header("Register Your Profile")
        name = st.text_input("Name")
        relative_name = st.text_input("Relative's Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        phone = st.text_input("Relative's Phone Number")

        # Add a new option for uploading or taking a photo
        photo_option = st.radio("Choose Photo Upload Method", ["Upload a Photo", "Take Screenshot with Camera"])

        if photo_option == "Upload a Photo":
            uploaded_photo = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"])
            if uploaded_photo:
                image_path = f"photos/{name}.jpg"
                os.makedirs("photos", exist_ok=True)
                image = Image.open(uploaded_photo)
                image.save(image_path)
        elif photo_option == "Take Screenshot with Camera":
            camera_photo = st.camera_input("Take a Photo")
            if camera_photo:
                image_path = f"photos/{name}.jpg"
                os.makedirs("photos", exist_ok=True)
                image = Image.open(camera_photo)
                image.save(image_path)
                
        col1, col2, col3, col4 = st.columns([4, 1, 1, 4])  # Assigning equal widths to columns
        with col2:
            if st.button("Register"):
                if name and relative_name and age and phone and image_path:
                    save_profile(name, relative_name, age, phone, image_path)
                    st.success("Profile Registered Successfully!")
                else:
                    st.error("Please fill all fields and upload a photo or take a screenshot.")
        with col3:
            try:
                if st.button("Clear Data"):
                    os.remove(PROFILE_FILE)  # Delete the profile file
                    st.success("Profile data cleared successfully!")
            except FileNotFoundError:
                st.warning("No profile data to clear.")
