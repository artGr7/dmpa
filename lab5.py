import cv2
import numpy as np


video = cv2.VideoCapture("ЛР4_main_video.mov",cv2.CAP_ANY)

def detect(kernel_size, standard_deviation, delta_tresh, min_area):

    _, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), standard_deviation)

    w = int(video.get(3))
    h = int(video.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('output.mp4', fourcc, 25, (w, h))

    while True:
        old_img = img.copy()
        flag, frame = video.read()
        if not flag:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(
            img, (kernel_size, kernel_size), standard_deviation)

        diff = cv2.absdiff(img, old_img)
        thresh = cv2.threshold(diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]
        contors, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contr in contors:
            if cv2.contourArea(contr) >= min_area:
                video_writer.write(frame)
                break
    video_writer.release()


kernel_size = 3
standard_deviation = 50
delta_tresh = 60
min_area = 20
detect(kernel_size, standard_deviation, delta_tresh, min_area)
