from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
from flask import Response, Flask, render_template
import threading
import pyglet
import imutils
import playsound
import time
import dlib
import cv2


outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

 
def sound_alarm():
    playsound.playsound('./fatigue/sounds/alarm2.mp3')

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


def generate():

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./fatigue/shape_predictor_68_face_landmarks.dat")


    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48

    COUNTER = 0
    ALARM_ON = False

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:

                    if not ALARM_ON:
                        ALARM_ON = True

                        t = Thread(target=sound_alarm)
                        t.deamon = True
                        t.start()

                    cv2.putText(frame, "WAKE UP!!!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                COUNTER = 0
                ALARM_ON = False

            cv2.putText(frame, "EAR: {:.4f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord("q"):
            break

        global outputFrame, lock
        
        with lock:
            outputFrame = frame.copy()
        
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


# def generate():
    
    # global outputFrame, lock
    # while True:
    #     with lock:
    #         if outputFrame is None:
    #             continue

    #         (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

    #         if not flag:
    #             continue

    #     yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # t = threading.Thread(target=detect_motion, args=(32,))
    # t.daemon = True
    # t.start()

    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True, use_reloader=False)
