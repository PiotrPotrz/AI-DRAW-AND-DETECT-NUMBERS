import cv2
import mediapipe as mp
import copy
import model
import warnings

warnings.filterwarnings("ignore")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6,
                       max_num_hands=1)

if cap.isOpened == False:
    print("Camera did not open")

drawing_list = []
prediction = None
while True:
    draw = False
    key = cv2.waitKey(1) & 0xFF
    ret, frame = cap.read()

    if key == ord("d"):
        draw = True

    if not ret:
        continue

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    height, width, chanels = frame.shape

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx > 8:
                    break
                cord_x = int(landmark.x * width)
                cord_y = int(landmark.y * height)
                if idx == 8 and cord_x >= 10 and cord_x <= 250 and cord_y >= 10 and cord_y <= 250:
                    cv2.circle(frame, (cord_x, cord_y), 20, (255, 255, 0), cv2.FILLED)
                    if draw == True:
                        drawing_list.append([cord_x, cord_y])

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    top_point = (10, 10)
    bottom_point = (250, 250)
    frame = cv2.rectangle(frame, top_point, bottom_point, (0, 0, 255), 1)

    if key == ord("k"):
        drawing_list = []

    if key == ord("v"):
        dw = copy.deepcopy(drawing_list)
        number2 = model.preprocessing(dw)
        prediction = model.predict(number2)

    for cords in drawing_list:
        frame = cv2.circle(frame, cords, 10, (0, 255, 0), cv2.FILLED)

    frame = cv2.flip(frame, 1)

    if prediction != None:
        frame = cv2.putText(frame, "Detected: " + str(prediction), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('DRAW AND DETECT', frame)

    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

