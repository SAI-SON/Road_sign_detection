import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.title("Road Sign Detection App")
st.write("Detect speed limit signs in images, videos, or with live camera")

# Load cascade classifiers
@st.cache_resource
def load_classifiers():
    classifiers = {
        "20km": cv2.CascadeClassifier("sign xml/20kmsign.xml"),
        "30km": cv2.CascadeClassifier("sign xml/30kmsign.xml"),
        "40km": cv2.CascadeClassifier("sign xml/cascade 40 km.xml"),
        "50km": cv2.CascadeClassifier("sign xml/cascade50km.xml"),
        "60km": cv2.CascadeClassifier("sign xml/cascade 60 km.xml"),
        "70km": cv2.CascadeClassifier("sign xml/cascade70km.xml"),
        "80km": cv2.CascadeClassifier("sign xml/cascade80km.xml"),
        "90km": cv2.CascadeClassifier("sign xml/cascade90km.xml"),
        "100km": cv2.CascadeClassifier("sign xml/cascade100km.xml")
    }
    return classifiers

classifiers = load_classifiers()

# Define detection parameters
detection_params = {
    "20km": {"scaleFactor": 1.3, "minNeighbors": 3, "color": (255, 0, 0)},    # Blue
    "30km": {"scaleFactor": 1.3, "minNeighbors": 3, "color": (0, 255, 0)},   # Green
    "40km": {"scaleFactor": 1.3, "minNeighbors": 4, "color": (0, 0, 255)},   # Red
    "50km": {"scaleFactor": 1.3, "minNeighbors": 4, "color": (255, 0, 255)}, # Magenta
    "60km": {"scaleFactor": 1.3, "minNeighbors": 3, "color": (0, 255, 255)},  # Yellow
    "70km": {"scaleFactor": 1.3, "minNeighbors": 3, "color": (255, 255, 0)}, # Cyan
    "80km": {"scaleFactor": 1.3, "minNeighbors": 4, "color": (128, 0, 128)}, # Purple
    "90km": {"scaleFactor": 1.3, "minNeighbors": 3, "color": (0, 128, 128)},  # Teal
    "100km": {"scaleFactor": 1.3, "minNeighbors": 3, "color": (255, 165, 0)} # Orange
}

# Set up sidebar controls
st.sidebar.header("Detection Settings")
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 4)
scale_factor = st.sidebar.slider("Scale Factor", 1.1, 1.5, 1.2, 0.05)

# Select which signs to detect
st.sidebar.header("Signs to Detect")
signs_to_detect = {}
for sign in classifiers.keys():
    signs_to_detect[sign] = st.sidebar.checkbox(f"{sign}/h", value=True)

# Function to detect signs
def detect_signs(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Make a copy of the original image for drawing
    result_img = image.copy()
    
    # Store detection results
    results = {}
    
    # Detect signs
    for sign, classifier in classifiers.items():
        if signs_to_detect[sign]:
            params = detection_params[sign]
            # Use either default parameters or user-adjusted ones
            sf = scale_factor if st.session_state.get(f"custom_sf_{sign}", False) else params["scaleFactor"]
            mn = min_neighbors if st.session_state.get(f"custom_mn_{sign}", False) else params["minNeighbors"]
            
            # Detect signs
            signs = classifier.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn)
            
            # Draw rectangles and labels
            for (x, y, w, h) in signs:
                cv2.rectangle(result_img, (x, y), (x + w, y + h), params["color"], 2)
                cv2.putText(result_img, f"{sign}/h", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, params["color"], 2)
            
            # Store results
            results[sign] = len(signs)
    
    return result_img, results

# Initialize session state for custom parameters
for sign in classifiers.keys():
    if f"custom_sf_{sign}" not in st.session_state:
        st.session_state[f"custom_sf_{sign}"] = False
    if f"custom_mn_{sign}" not in st.session_state:
        st.session_state[f"custom_mn_{sign}"] = False

# Custom parameters checkboxes
st.sidebar.header("Custom Parameters")
for sign in classifiers.keys():
    st.sidebar.checkbox(f"Custom SF for {sign}", key=f"custom_sf_{sign}")
    st.sidebar.checkbox(f"Custom MN for {sign}", key=f"custom_mn_{sign}")

# Select input mode
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio("Select Input Mode", ["Live Camera", "Upload Image", "Upload Video", "Take Snapshot"])

# Live Camera Section
if input_mode == "Live Camera":
    st.header("Live Camera Feed")
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    # Live camera detection
    class VideoProcessor:
        def __init__(self):
            self.results = {}
            self.frame_count = 0
            self.process_every_n_frames = 3  # Process every 3rd frame
            self.frame_size = (640, 480)  # Reduced resolution for faster processing
            
        def recv(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            # Resize frame for faster processing
            img = cv2.resize(img, self.frame_size)
            
            # Only process every nth frame
            if self.frame_count % self.process_every_n_frames == 0:
                # Convert to grayscale once
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Process frame
                for sign, classifier in classifiers.items():
                    if signs_to_detect[sign]:
                        params = detection_params[sign]
                        # Adjust parameters for faster detection
                        signs = classifier.detectMultiScale(
                            gray,
                            scaleFactor=params["scaleFactor"],
                            minNeighbors=params["minNeighbors"],
                            minSize=(30, 30),
                            maxSize=(150, 150)
                        )
                        
                        # Draw rectangles and labels
                        for (x, y, w, h) in signs:
                            cv2.rectangle(img, (x, y), (x + w, y + h), params["color"], 2)
                            cv2.putText(img, f"{sign}/h", (x, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, params["color"], 2)
                            
                        self.results[sign] = len(signs)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # Create WebRTC streamer
    ctx = webrtc_streamer(
        key="road-sign-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display help text
    st.info("The live camera feed will detect speed limit signs in real-time. Adjust detection parameters in the sidebar.")

# Image Upload Section
elif input_mode == "Upload Image":
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # For image files
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # Convert RGB to BGR (OpenCV format)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect signs
        result_img, results = detect_signs(img_array)
        
        # Convert back to RGB for display
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Display the original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Processed Image")
            st.image(result_img, use_column_width=True)
        
        # Display detection results
        st.subheader("Detection Results")
        for sign, count in results.items():
            if count > 0:
                st.write(f"{sign}/h sign: {count} detected")

# Camera Snapshot Section
elif input_mode == "Take Snapshot":
    st.header("Take a Snapshot")
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Process the snapshot
        image = Image.open(camera_image)
        img_array = np.array(image.convert('RGB'))
        
        # Convert RGB to BGR (OpenCV format)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect signs
        result_img, results = detect_signs(img_array)
        
        # Convert back to RGB for display
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Display processed image
        st.subheader("Processed Image")
        st.image(result_img, use_column_width=True)
        
        # Display detection results
        st.subheader("Detection Results")
        for sign, count in results.items():
            if count > 0:
                st.write(f"{sign}/h sign: {count} detected")

# Video Upload Section
elif input_mode == "Upload Video":
    st.header("Upload Video")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        # Process video
        video = cv2.VideoCapture(tfile.name)
        
        # Create a placeholder for the processed video
        video_placeholder = st.empty()
        
        # Get video properties
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a progress bar
        progress_bar = st.progress(0)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process 1 frame every N frames to speed up processing
        process_every_n_frames = 5
        
        if st.button("Process Video"):
            current_frame = 0
            
            while video.isOpened():
                ret, frame = video.read()
                
                if not ret:
                    break
                
                current_frame += 1
                
                # Update progress
                progress_bar.progress(min(current_frame / frame_count, 1.0))
                
                # Process only every Nth frame
                if current_frame % process_every_n_frames == 0:
                    # Process frame
                    processed_frame, _ = detect_signs(frame)
                    
                    # Convert to RGB for display
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the processed frame
                    video_placeholder.image(processed_frame_rgb, caption="Processed Video", use_column_width=True)
            
            # Release video resources
            video.release()
            
            st.success("Video processing complete!")

# Add installation instructions
st.sidebar.markdown("---")
st.sidebar.subheader("Installation")
st.sidebar.info(
    """
    If you haven't installed required packages yet, run:
    
    ```
    pip install streamlit opencv-python pillow numpy streamlit-webrtc
    ```
    
    For webcam support:
    ```
    pip install av aiortc
    ```
    """
)

# Add helpful information
st.sidebar.markdown("---")
st.sidebar.subheader("Help")
st.sidebar.info(
    """
    This app detects speed limit signs using various input methods:
    
    - Live Camera: Real-time sign detection
    - Upload Image: Process a single image file
    - Take Snapshot: Capture and process a photo
    - Upload Video: Process a video file
    
    Adjust detection parameters in the sidebar for better results.
    """
)

# Footer
st.markdown("---")
st.caption("Road Sign Detection App using Haar Cascade Classifiers")