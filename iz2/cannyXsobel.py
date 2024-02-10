import cv2

image = cv2.imread('media/d.jpg', cv2.IMREAD_GRAYSCALE)

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