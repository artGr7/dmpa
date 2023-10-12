import cv2

# Открываем камеру
cap = cv2.VideoCapture(0)  # Обычно 0 означает встроенную камеру

# Проверяем, успешно ли открылась камера
if not cap.isOpened():
    print("Ошибка: Камера не может быть открыта.")
    exit()

# Получаем параметры видеопотока
frame_width = int(cap.get(3) )  # Ширина кадра
frame_height = int(cap.get(4))  # Высота кадра

# Создаем объект VideoWriter для записи видео в файл в формате MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Захватываем видеопоток с камеры, записываем в файл и демонстрируем
while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка: Невозможно получить кадр с камеры.")
        break


    # Записываем кадр в файл
    out.write(frame)

    # Отображаем кадр с камеры
    cv2.imshow('Camera', frame)

    # Для выхода нажмите клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы, закрываем окна и файл записи
cap.release()
out.release()
cv2.destroyAllWindows()

# Демонстрируем записанное видео
cap2 = cv2.VideoCapture('output.mp4')
while cap2.isOpened():
    ret, frame = cap2.read()

    if not ret:
        break

    cv2.imshow('Recorded Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap2.release()
cv2.destroyAllWindows()