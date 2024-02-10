import cv2
import numpy as np

image = cv2.imread('media/d.jpg', cv2.IMREAD_GRAYSCALE)

# Перебор значений параметров размытия Гаусса
for sigma in [1, 5, 9]:
    # Размытие Гаусса
    blurred_image = cv2.GaussianBlur(image, (sigma, sigma), 0)

    # Перебор значений пороговых значений для алгоритма Канни
    for low_threshold in [10, 30, 50]:
        high_threshold = low_threshold + 20

        # Оператор Робертса
        roberts_x = cv2.filter2D(blurred_image, -1, np.array([[1, 0], [0, -1]]))
        roberts_y = cv2.filter2D(blurred_image, -1, np.array([[0, 1], [-1, 0]]))

        # Длина вектора
        roberts_combined = np.sqrt(np.square(roberts_x) + np.square(roberts_y))

        # Алгоритм Канни с оператором Робертса
        canny_edges = cv2.Canny(np.uint8(roberts_combined), low_threshold, high_threshold)
        resized_image = cv2.resize(canny_edges, (400, 300))
        cv2.imshow(f'(Sigma={sigma}, Low={low_threshold}, High={high_threshold})', resized_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()