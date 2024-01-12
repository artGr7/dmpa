import cv2 as cv
import numpy as np
import math

def angles(x,y):
    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360

    octant = int((angle_deg + 22.5) % 360 / 45)
    return octant

def convolution(img,kernel):
    new_img = np.zeros_like(img, dtype=np.float32)
    w = img.shape[0]
    h = img.shape[1]

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            new_img[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel)

    return new_img


def main():
    img = cv.imread('cat.jpg',cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(600,540))
    img = cv.GaussianBlur(img, (7,7), 6)
    cv.imshow('cat',img)

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    img_Gx = convolution(img,Gx)
    img_Gy = convolution(img,Gy)

    grad_leng = np.sqrt(img_Gx**2 + img_Gy**2)

    grad_ang = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range (img.shape[1]):
            grad_ang[i][j] = angles(img_Gx[i][j],img_Gy[i][j])

    print(grad_leng)
    print(grad_ang)

    #####################

    border = np.zeros(img.shape)

    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            grad = grad_leng[i][j]
            direct = grad_ang[i][j]

            if (direct == 0 or direct == 4):
                x_shift = 0
            elif (direct > 0 and direct < 4):
                x_shift = 1
            else:
                x_shift = -1

            if (direct == 2 or direct == 6):
                y_shift = 0
            elif (direct > 2 and direct < 6):
                y_shift = -1
            else:
                y_shift = 1

            if grad >= grad_leng[i+x_shift][j+y_shift] and grad >= grad_leng[i-x_shift][j-y_shift]: 
                border[i][j]=255

    cv.imshow('border',border)

    ######################

    filter = np.zeros_like(img)
    max_grad = np.max(grad_leng)
    low_level = max_grad//50
    high_level = max_grad/5

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            grad = grad_leng[i][j]
            if border[i][j]==255:
                if grad > high_level:
                    filter[i][j] = 255
                elif grad >= low_level and grad <= high_level:
                    is_border = False
                    for k in range(-1,2):
                        for l in range(-1,2):
                            if border[i+k][j+l] == 255:
                                is_border = True
                                break
                        if is_border:
                            break
                    if is_border==False:
                        filter[i][j]=255

    cv.imshow('filtered border',filter)

    cv.waitKey(0)
    cv.destroyAllWindows()


main()
