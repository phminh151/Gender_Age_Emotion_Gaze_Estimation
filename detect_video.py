# A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import datetime
import dlib
# import numpy as np

# emotion detection
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
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


model = create_emotion_model("model.h5")
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


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frameHeight / 150)),
                8,
            )
    return frameOpencvDnn, faceBoxes


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

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

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
# video_path = "C:\\AI\\Data\\face_video\\production ID_4734773.mp4"
# video_path = "C:\\AI\\Data\\face_video\\video.mp4"
# video_path = "rtsp://admin:HDghn%40%23192khrp@113.161.43.181:555/Streaming/channels/1"
video_path = "C:\\Users\\Admin\\Downloads\\Video\\people.mp4"
video = cv2.VideoCapture(video_path)
padding = 20
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
        # print(datetime.datetime.now())
        cv2.imshow(
            "Detecting age and gender",
            cv2.resize(frame, (1080, 720), interpolation=cv2.INTER_CUBIC),
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for faceBox in faceBoxes:
        try:
            # # add emotion face
            start_time = datetime.datetime.now()
            x = faceBox[0]
            y = faceBox[1]
            w = faceBox[2] - faceBox[0]
            h = faceBox[3] - faceBox[1]
            roi_gray = gray[faceBox[1] : faceBox[3], faceBox[0] : faceBox[2]]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
            )
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(
                resultImg,
                f"{emotion_dict[maxindex]}",
                (faceBox[0], faceBox[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            end_time = datetime.datetime.now()
            print(f"Total time: {(end_time - start_time)}")
            # end emotion face

            face = frame[
                max(0, faceBox[1] - padding) : min(
                    faceBox[3] + padding, frame.shape[0] - 1
                ),
                max(0, faceBox[0] - padding) : min(
                    faceBox[2] + padding, frame.shape[1] - 1
                ),
            ]
            # print(face)

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
                resultImg,
                f"{gender}, {age}",
                (faceBox[0], faceBox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except:
            pass
        # print(datetime.datetime.now())
        cv2.imshow(
            "Detecting age and gender",
            cv2.resize(resultImg, (1080, 720), interpolation=cv2.INTER_CUBIC),
        )
