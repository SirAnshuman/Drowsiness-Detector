from itertools import count
import cv2
import numpy as np
import dlib
from imutils import face_utils
from playsound import playsound
from twilio.rest import Client
import keys
import time
import matplotlib.pyplot as plt

x_vals = []
y_vals = []
index = count()

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
new_time = 0
pre_time = 0
d = 0
s = 0
a = 0


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if (ratio > 0.25):
        return 2
    elif (ratio > 0.21 and ratio <= 0.25):
        return 1
    else:
        return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    faces_frame = frame.copy()

    new_time = time.time()
    fps = 1/(new_time - pre_time)
    pre_time = new_time
    fps = int(fps)
    cv2.putText(faces_frame, str(fps),(8, 80), cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,0),4)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(faces_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if (left_blink == 0 or right_blink == 0):
            s = s+1
            x_vals.append(next(index))
            y_vals.append("Sleeping")
            sleep += 1
            drowsy = 0
            active = 0

            if (sleep > 500):
                client = Client(keys.account_sid, keys.auth_token)
                message = client.messages.create(
                    body="Anshuman Tripathi has fallen asleep in your class."
                         "  Enrollment No - 0002CB201013"
                         "  Semester - 6th",
                    from_=keys.twilio_number,
                    to=keys.target_number
                )

            if (sleep > 6):
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                if(sleep > 100):
                    playsound('Alarm.mp3')


        elif (left_blink == 1 or right_blink == 1):
            d = d+1
            x_vals.append(next(index))
            y_vals.append("Drowsy")
            sleep = 0
            active = 0
            drowsy += 1
            if (drowsy > 6):
                status = "Drowsy !"
                color = (0, 0, 255)

        else:
            a = a+1
            x_vals.append(next(index))
            y_vals.append("Active")
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 6):
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(faces_frame, (x, y), 1, (255, 255, 255), -1)



    cv2.imshow("Frame", frame)
    cv2.imshow("Detector Output", faces_frame)

    key = cv2.waitKey(1)
    if key == 27:
        Status = 'Sleepy', 'Drowsy', 'Active'
        Time = [s, d, a]
        explode = (0, 0, 0.1)

        plt.subplot(121)
        plt.plot(x_vals, y_vals)

        plt.subplot(122)
        plt.pie(Time, explode=explode, labels=Status, autopct='%1.2f%%')

        plt.suptitle('Anshuman Tripathi MindMonitor Data')
        plt.savefig("chart.jpg")

        plt.show()
        break
