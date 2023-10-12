import cv2
import numpy as np

# Открываем камеру. Обычно 0 означает встроенную камеру
cap = cv2.VideoCapture(0)

# Захватываем видеопоток с камеры
while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка: Невозможно получить кадр с камеры.")
        break

    height, width, _ = frame.shape
    cross_size = 50
    cross_thickness = 2

    # Извлекаем цвет центрального пикселя
    center_pixel_color = frame[height // 2, width // 2]

    closest_color = np.argmax(center_pixel_color)

    # Заполняем крест выбранным цветом
    if closest_color == 0:
        cross_color = (255, 0, 0)  # Красный
    elif closest_color == 1:
        cross_color = (0, 255, 0)  # Зеленый
    else:
        cross_color = (0, 0, 255)  # Синий

    cv2.rectangle(frame, (width // 2 - 110, height // 2 - 10), (width // 2 + 110, height // 2 + 10), cross_color, cross_thickness)
    cv2.rectangle(frame, (width // 2 - 10, height // 2 - 110), (width // 2 + 10, height // 2 - 10), cross_color, cross_thickness)
    cv2.rectangle(frame, (width // 2 - 10, height // 2 + 110), (width // 2 + 10, height // 2 + 10), cross_color, cross_thickness)

    # Отображаем кадр
    cv2.imshow('Camera', frame)

    # Для выхода нажмите клавишу 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Освобождаем ресурсы и закрываем окно
cap.release()
cv2.destroyAllWindows()