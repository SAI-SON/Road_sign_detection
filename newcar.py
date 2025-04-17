import cv2

# Load the cascade files for each speed limit sign
alg_30km = "D:/Road_sign_board_detection/sign xml/30kmsign.xml"
alg_20km = "D:/Road_sign_board_detection/sign xml/20kmsign.xml"
alg_40km = "D:/Road_sign_board_detection/sign xml/cascade 40 km.xml"
alg_50km = "D:/Road_sign_board_detection/sign xml/cascade50km.xml"
alg_60km = "D:/Road_sign_board_detection/sign xml/cascade 60 km.xml"
alg_70km = "D:/Road_sign_board_detection/sign xml/cascade70km.xml"
alg_80km = "D:/Road_sign_board_detection/sign xml/cascade80km.xml"
alg_90km = "D:/Road_sign_board_detection/sign xml/cascade90km.xml"
alg_100km = "D:/Road_sign_board_detection/sign xml/cascade100km.xml"

haar_cascade_30km = cv2.CascadeClassifier(alg_30km)
haar_cascade_20km = cv2.CascadeClassifier(alg_20km)
haar_cascade_40km = cv2.CascadeClassifier(alg_40km)
haar_cascade_50km = cv2.CascadeClassifier(alg_50km)

haar_cascade_60km = cv2.CascadeClassifier(alg_60km)
haar_cascade_70km = cv2.CascadeClassifier(alg_70km)
haar_cascade_80km = cv2.CascadeClassifier(alg_80km)
haar_cascade_90km = cv2.CascadeClassifier(alg_90km)
haar_cascade_100km = cv2.CascadeClassifier(alg_100km)

# Variables for accuracy calculation
total_frames = 0
correct_detections_30km = 0
correct_detections_20km = 0
correct_detections_40km = 0
correct_detections_50km = 0
correct_detections_60km = 0
correct_detections_70km = 0
correct_detections_80km = 0
correct_detections_90km = 0
correct_detections_100km = 0
false_detections = 0

# Specify which signs are present for testing purposes
sign_30km_present_in_frame = True
sign_20km_present_in_frame = False
sign_40km_present_in_frame = False
sign_50km_present_in_frame = False
sign_60km_present_in_frame = False
sign_70km_present_in_frame = False
sign_80km_present_in_frame = False
sign_90km_present_in_frame = False
sign_100km_present_in_frame = False

# Capture video from the webcam
cam = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cam.read()
    total_frames += 1
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect and annotate each sign
    signs_30km = haar_cascade_30km.detectMultiScale(grayImg, scaleFactor=1.25, minNeighbors=4)
    signs_20km = haar_cascade_20km.detectMultiScale(grayImg, scaleFactor=1.4, minNeighbors=4)
    signs_40km = haar_cascade_40km.detectMultiScale(grayImg, scaleFactor=1.21, minNeighbors=7)
    signs_50km = haar_cascade_50km.detectMultiScale(grayImg, scaleFactor=1.25, minNeighbors=5)
    signs_60km = haar_cascade_60km.detectMultiScale(grayImg, scaleFactor=1.2, minNeighbors=4)
    signs_70km = haar_cascade_70km.detectMultiScale(grayImg, scaleFactor=1.18, minNeighbors=4)
    signs_80km = haar_cascade_80km.detectMultiScale(grayImg, scaleFactor=1.22, minNeighbors=5)
    signs_90km = haar_cascade_90km.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=2)
    signs_100km = haar_cascade_100km.detectMultiScale(grayImg, scaleFactor=1.19, minNeighbors=4)

    # Display and count detections
    if len(signs_30km) > 0:
        correct_detections_30km += 1 if sign_30km_present_in_frame else 0
        false_detections += 1 if not sign_30km_present_in_frame else 0
        for (x, y, w, h) in signs_30km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(img, "30 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if len(signs_20km) > 0:
        correct_detections_20km += 1 if sign_20km_present_in_frame else 0
        false_detections += 1 if not sign_20km_present_in_frame else 0
        for (x, y, w, h) in signs_20km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.putText(img, "20 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if len(signs_40km) > 0:
        correct_detections_40km += 1 if sign_40km_present_in_frame else 0
        false_detections += 1 if not sign_40km_present_in_frame else 0
        for (x, y, w, h) in signs_40km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(img, "40 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if len(signs_50km) > 0:
        correct_detections_50km += 1 if sign_50km_present_in_frame else 0
        false_detections += 1 if not sign_50km_present_in_frame else 0
        for (x, y, w, h) in signs_50km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            cv2.putText(img, "50 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    if len(signs_60km) > 0:
        correct_detections_60km += 1 if sign_60km_present_in_frame else 0
        false_detections += 1 if not sign_60km_present_in_frame else 0
        for (x, y, w, h) in signs_60km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 4)
            cv2.putText(img, "60 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if len(signs_70km) > 0:
        correct_detections_70km += 1 if sign_70km_present_in_frame else 0
        false_detections += 1 if not sign_70km_present_in_frame else 0
        for (x, y, w, h) in signs_70km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 4)
            cv2.putText(img, "70 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    if len(signs_80km) > 0:
        correct_detections_80km += 1 if sign_80km_present_in_frame else 0
        false_detections += 1 if not sign_80km_present_in_frame else 0
        for (x, y, w, h) in signs_80km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 128), 4)
            cv2.putText(img, "80 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 128), 2)

    if len(signs_90km) > 0:
        correct_detections_90km += 1 if sign_90km_present_in_frame else 0
        false_detections += 1 if not sign_90km_present_in_frame else 0
        for (x, y, w, h) in signs_90km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 128), 4)
            cv2.putText(img, "90 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 128), 2)

    if len(signs_100km) > 0:
        correct_detections_100km += 1 if sign_100km_present_in_frame else 0
        false_detections += 1 if not sign_100km_present_in_frame else 0
        for (x, y, w, h) in signs_100km:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 165, 0), 4)
            cv2.putText(img, "100 km/h sign detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)

    # Display the image with detected signs
    cv2.imshow("Roadsign", img)

    # Exit on pressing ESC
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all windows
cam.release()
cv2.destroyAllWindows()
