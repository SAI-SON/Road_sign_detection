# Road Sign Detection Application

A real-time application to detect speed limit signs using computer vision and Haar cascade classifiers.

## Features

- **Live Camera Detection**: Real-time detection of speed limit signs through webcam
- **Image Upload**: Analyze speed limit signs in static images
- **Video Processing**: Process video files to identify speed limit signs
- **Camera Snapshot**: Take and analyze a photo instantly
- **Customizable Detection Parameters**: Adjust sensitivity settings for better detection

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/road-sign-detection.git
cd road-sign-detection
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

### Additional Requirements

- Make sure your webcam is properly connected and configured
- The application expects the XML cascade files to be in a folder named "sign xml" in the same directory as the application

## Usage

1. Run the Streamlit application:
```
streamlit run stream_roadsign.py
```

2. The application will launch in your default web browser
3. Select your preferred input mode from the sidebar:
   - **Live Camera**: For real-time detection
   - **Upload Image**: To analyze static images
   - **Upload Video**: To process video files
   - **Take Snapshot**: To capture and analyze a photo

4. Adjust detection parameters in the sidebar for optimal results

## Detection Parameters

- **Scale Factor**: Controls the step size when scaling the image (default: 1.3)
- **Min Neighbors**: Determines how many neighbors each candidate rectangle should have (default varies by sign)
- **Custom Parameters**: You can customize parameters for each sign type individually

## Supported Speed Limit Signs

- 20 km/h
- 30 km/h
- 40 km/h
- 50 km/h
- 60 km/h
- 70 km/h
- 80 km/h
- 90 km/h
- 100 km/h

## Troubleshooting

- If you encounter webcam access issues, make sure your browser has permission to access the camera
- For performance issues with live detection, try:
  - Reducing the number of signs to detect simultaneously
  - Adjusting the processing frequency in the VideoProcessor class
  - Using a smaller frame size

## License

This project is licensed under the MIT License - see the LICENSE file for details. 