import cv2
import numpy as np

# Задание 1: Чтение изображения с камеры и перевод в формат HSV
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow('HSV Image', hsv_image)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Задание 2: Фильтрация изображения и вывод красной части
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])  # Нижний диапазон для красного
    upper_red = np.array([10, 255, 255])  # Верхний диапазон для красного

    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    result = cv2.bitwise_and(frame, frame, mask=red_mask)

    cv2.imshow('Red Threshold', result)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Задание 3: Морфологические преобразования
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    white_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    red_mask = cv2.bitwise_and(frame, frame, mask=white_mask)

    # Применение морфологических операций
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Задание 4: Нахождение моментов и площади объекта
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    moments = cv2.moments(red_mask)
    area = moments['m00']

    if area > 0:
        cx = int(moments['m10'] / area)
        cy = int(moments['m01'] / area)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
