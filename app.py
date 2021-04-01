"""

    Created on Sun Dec 2 20:54:11 2018
    
    @author: keyur-r

    SSD pretrained caffe model based face detection using it with opencv's dnn module.
    (https://docs.opencv.org/3.4.0/d5/de7/tutorial_dnn_googlenet.html)
    
    python face_detection_ssd.py -p <prototxt> -m <caffe-model> -t <thresold>

"""

from imutils import face_utils, video
import dlib
import cv2
import argparse
import os
import numpy as np
import datetime

# import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from gaze_tracking import GazeTracking
from flask import Flask, render_template, Response
app = Flask(__name__)
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_emotion_model(model_path):
    # Create the model
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1))
    )
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))

    model.load_weights(model_path)

    return model


model = create_emotion_model("emotion_model\model.h5")
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

ageProto = "age_model/age_deploy.prototxt"
ageModel = "age_model/age_net.caffemodel"
genderProto = "gender_model\gender_deploy.prototxt"
genderModel = "gender_model\gender_net.caffemodel"


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
genderList = ["Male", "Female"]

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 20

def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    """
    To draw some fancy box around founded faces in stream
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

gaze = GazeTracking()

def detect_eyes(face):    
    # We send this frame to GazeTracking to analyze it    
    gaze.refresh(face)
    
    face = gaze.annotated_frame()
    
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    # cv2.putText(face, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # left_pupil = gaze.pupil_left_coords()
    # right_pupil = gaze.pupil_right_coords()
    # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    return face, text


def find_faces(img, detections, gray):
    total_faces = 0
    start = datetime.datetime.now()
        # Draw boxes around found faces
    for i in range(0, detections.shape[2]):
        try:
            # Probability of prediction
            prediction_score = detections[0, 0, i, 2]
            if prediction_score < args.thresold:
                continue
            # Finding height and width of frame
            (h, w) = img.shape[:2]
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            total_faces += 1

            prediction_score_str = "{:.2f}%".format(prediction_score * 100)

            label = "Face #{} ({})".format(total_faces, prediction_score_str)

            # https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
            draw_fancy_box(img, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)
            # show the face number with prediction score
            # cv2.putText(
            #     img,
            #     label,
            #     (x1 - 20, y1 - 20),
            #     cv2.FONT_HERSHEY_TRIPLEX,
            #     0.6,
            #     (51, 51, 255),
            #     2,
            # )
                    
            # Age, gender

            face = img[
                max(0, y1 - padding) : min(
                    y2 + padding, img.shape[0] - 1
                ),
                max(0, x1 - padding) : min(
                    x2 + padding, img.shape[1] - 1
                ),
            ]

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
            )

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # print(f"Gender: {gender}")

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            # print(f"Age: {age[1:-1]} years")

            cv2.putText(
                img,
                f"{gender}, {age}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # # add emotion face
            # start_time = datetime.datetime.now()
            roi_gray = gray[y1: y2, x1: x2]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
            )
            prediction = model.predict(cropped_img)
            for index, value in emotion_dict.items():
                    emotion_label = value
                    emotion_score = prediction[0][index]
                    bar_x = 50 #this is the size if an emotion is 100%
                    bar_x = int(bar_x * emotion_score)

                    text_location_y = y1 + 20 + (index+1) * 25
                    text_location_x = x1+(x2-x1)+10
                    
                    cv2.putText(img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.rectangle(img
                        , (x2+110, text_location_y -12)
                        , (x2+110+50, text_location_y -2)
                        , (255,255,255), 2)
                    cv2.rectangle(img
                        , (x2+110, text_location_y -12)
                        , (x2+110+bar_x, text_location_y -2)
                        , (0,0,255), cv2.FILLED)
            maxindex = int(np.argmax(prediction))
            cv2.putText(
                img,
                f"{emotion_dict[maxindex]}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # end_time = datetime.datetime.now()
            # print(f"Total time: {(end_time - start_time)}")
            # end emotion face

            # Eyes
            eyes, eye_direction = detect_eyes(img[y1 : y2, x1 : x2])
            img[y1 : y2, x1 : x2] = eyes
            cv2.putText(
                img,
                f"{eye_direction}",
                (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            # DETECT MOTION
            # start_time = datetime.datetime.now()
            
        except:
            pass
    
    end = datetime.datetime.now()
    print(f"Total faces: {total_faces} with {end - start}")
    # show the output frame
    # cv2.imshow("Face Detection with SSD", img)
    return img, age, gender
    


def face_detection_realtime():

    # Feed from computer camera with threading
    # total = video.count_frames(video_patpyth)
    cap = video.VideoStream(0).start()
    # cap = cv2.VideoCapture(video_path)

    count = 0
    while True:
        try:
            count += 1
            print(f"{count}")
            # Getting out image frame by webcam
            # _, img = cap.read()
            img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
            inputBlob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 1, (300, 300), (104, 177, 123)
            )

            detector.setInput(inputBlob)
            detections = detector.forward()
            img, age, gender = find_faces(img, detections, gray)
            
            # return stream video to client
            imgencode=cv2.imencode('.jpg',img)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            yield age
            yield gender
            # cv2.imshow('img', img)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
        except:
            pass
    del(cap)

    cv2.destroyAllWindows()
    cap.stop()


@app.route('/vid')
def vid():
    return Response(face_detection_realtime(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p",
        "--prototxt",
        default="./deploy.prototxt.txt",
        help="Caffe 'deploy' prototxt file",
    )
    ap.add_argument(
        "-m",
        "--model",
        default="face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        help="Pre-trained caffe model",
    )
    ap.add_argument(
        "-t",
        "--thresold",
        type=float,
        default=0.6,
        help="Thresold value to filter weak detections",
    )
    args = ap.parse_args()

    # This is based on SSD deep learning pretrained model
    detector = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    print("Real time face detection is starting ... ")
    face_detection_realtime()
    app.run(host='localhost',port=5000, debug=True, threaded=True)
