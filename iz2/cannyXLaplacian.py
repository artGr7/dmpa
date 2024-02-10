import cv2
import numpy as np

image = cv2.imread('media/d.jpg', cv2.IMREAD_GRAYSCALE)

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