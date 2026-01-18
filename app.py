import threading
import numpy as np
import cv2
from deepface import DeepFace

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
face_match = False
# Ensure this path is correct
ref_img = cv2.imread("static/reference.jpg") 

def check_face(frame):
    global face_match
    try:
        # DeepFacqe.verify returns a dict; check the 'verified' key
        result = DeepFace.verify(frame, ref_img.copy(), enforce_detection=False)
        face_match = result['verified']
    except Exception as e:
        face_match = False

while True:
    ret, frame = cap.read()
    
    if ret:
        # Run verification every 30 frames to avoid lag
        if count % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()
        
        count += 1 # Corrected increment

        # Visual feedback
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    # Press 'q' to quit
    key = cv2.waitKey(1) # Changed from 0 to 1 for live video
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
