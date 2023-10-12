import cv2

image = cv2.imread('media/123.jpg')

# Перевести изображение в формат HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Создать окна для вывода изображений
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)

# Вывести изображения в окна
cv2.imshow('Original Image', image)
cv2.imshow('HSV Image', hsv_image)

cv2.waitKey(0)
cv2.destroyAllWindows()