import cv2
import numpy as np

image = cv2.imread('d.jpg')

# Перебор значений параметров размытия Гаусса
for sigma in [1, 5, 9]:
    # Размытие Гаусса
    blurred_image = cv2.GaussianBlur(image, (sigma, sigma), 0)

    # Перебор значений пороговых значений для алгоритма Канни
    for low_threshold in [50, 100, 150]:
        high_threshold = low_threshold + 50
        # Оператор Лапласа
        laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

        # Применение алгоритма Canny с оператором Лапласа
        canny_edges = cv2.Canny(np.uint8(np.absolute(laplacian)), low_threshold, high_threshold)
        resized_image = cv2.resize(canny_edges, (400, 300))
        cv2.imshow(f'(Sigma={sigma}, Low={low_threshold}, High={high_threshold})', resized_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

//

import cv2
import numpy as np

image = cv2.imread('d.jpg')

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

//

import cv2

image = cv2.imread('d.jpg')

# Перебор значений параметров размытия Гаусса
for sigma in [1, 5, 15]:
    # Размытие Гаусса
    blurred_image = cv2.GaussianBlur(image, (sigma, sigma), 0)

    # Перебор значений пороговых значений для алгоритма Canny
    for low_threshold in [50, 100, 150]:
        high_threshold = low_threshold + 50

        # Алгоритм Канни с оператором Собеля
        canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
        resized_image = cv2.resize(canny_edges, (400, 300))
        cv2.imshow(f'(Sigma={sigma}, Low={low_threshold}, High={high_threshold})', resized_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
