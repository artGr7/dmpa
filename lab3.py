import cv2 as cv
import numpy


def kern(x, y, sigma, a, b):
    e = numpy.exp(-(((x-a)**2+(y-b)**2)/(2*sigma**2)))
    return 1/(2*numpy.pi*sigma**2)*e

def Gauss(img, size, dev):
    kernel = numpy.zeros((size, size))
    a = int(size/2)+1
    val = 0
    for i in range(size):
        for j in range(size):
            kernel[i, j] = kern(i, j, dev, a, a)
    print(kernel, '\n')
    sum = numpy.sum(kernel)
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= sum
    print(kernel, '\n')
    copy = img.copy()
    sum = numpy.sum(kernel)
    print(sum, '\n')
    for i in range((size//2), img.shape[0]-(size//2)):
        for j in range((size//2), img.shape[1]-(size//2)):
            for k in range(-(size//2), (size//2)+1):
                for l in range(-(size//2), (size//2)+1):
                    val += img[i+k][j+l]*kernel[(size//2)+k][(size//2)+l]
            copy[i][j] = val
    print(copy)
    return copy


def main():
    img = cv.imread(r'media/941898.jpg', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (300, 300))

    cv.imshow('original picture', img)

    blur = Gauss(img, 5, 3)
    cv.imshow('blured picture', blur)

    blur_opencv = cv.GaussianBlur(img, (5, 5), 3)
    cv.imshow('opencv blured picture', blur_opencv)

    cv.waitKey(0)
    
main()
