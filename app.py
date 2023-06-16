import csv
import itertools

import cv2
import numpy as np
import mediapipe as mp
from keypoint_classifier import KeyPointClassifier


def main():
    cap = cv2.VideoCapture(0)

    mp_drawing = mp.solutions.drawing_utils

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48

        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                if number != -1:
                    csv_path = 'keypoint_classifier/keypoint.csv'
                    with open(csv_path, 'a', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([number, *pre_processed_landmark_list])
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,)
                print(keypoint_classifier_labels[hand_sign_id])

        cv2.imshow('Hand Gesture Recognition', image)

    cap.release()
    cv2.destroyAllWindows()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(temp_landmark_list):
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = list(map(lambda n: n/max_value, temp_landmark_list))

    return temp_landmark_list

if __name__ == '__main__':
    main()
