import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/hafez/PycharmProjects/ImageProccessing_P2/test.jpg', 1)


# # 1 --- Show Image
def showImage():
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)

# # 2 --- Blue Image
def blueImage():
    img[:, :, 2] = 0

    cv2.imshow('Blue Image', img)
    cv2.waitKey(0)


# # 3 --- Displaying in gray scale color
def grayScale():
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
    plt.show()


# # 4 --- Smoothing with a Gaussian filter
def gaussianFilter():
    smoothed = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imshow('Original', img)
    cv2.imshow('Smoothed', smoothed)

    cv2.waitKey(0)


# # 5 --- Rotate image by 90 degrees
def rotateImage():
    height = img.shape[0]
    width = img.shape[1]

    mat = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
    dst = cv2.warpAffine(img, mat, (width, height))

    cv2.imshow('Rotated', dst)
    cv2.waitKey(0)


# # 6 --- resize image
def resizeImage():
    width = int(img.shape[1] * 50 / 100)
    height = int(img.shape[0])
    dim = (width, height)

    res = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)

    cv2.imshow('Resize', res)
    cv2.waitKey(0)


# # 7 --- Edge detection
def edgeDetection():
    edges = cv2.Canny(img,100,200)

    cv2.imshow('Edge Image', edges)
    cv2.waitKey(0)


# # 8 --- Segmentation
def segmentation():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('Segmentation', thresh)
    cv2.waitKey(0)

# # 9 --- Face detection with a rectangle around the faces
def faceDetection():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        '/home/hafez/PycharmProjects/ImageProccessing_P2/haarcascade_frontalface_alt.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5);

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Detection', img)
    cv2.waitKey(0)


# # 10 --- Show five frames of a given video
def videoFrames():
    cap = cv2.VideoCapture(
        '/home/hafez/PycharmProjects/ImageProccessing_P2/hooshyar_khayam.mp4')
    count = 0
    while(count < 10):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        count += 1

        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

showImage()
