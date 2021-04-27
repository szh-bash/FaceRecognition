import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace


sys.path.append('.')


def retina_face(img_path, img_result='test_detector.jpg'):
    thresh = 0.8
    scales = [250, 250]

    gpuid = -1
    detector = RetinaFace('/home/shenzhonghai/FaceClustering/detection/RetinaFace/model/R50', 0, gpuid, 'net3')
    # detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

    img = cv2.imread(img_path)
    # print(img.shape)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #im_scale = 1.0
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    # print('im_scale', im_scale)

    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(img,
                                       thresh,
                                       scales=scales,
                                       do_flip=flip)
    # print(landmarks)
    # print(c, faces.shape, landmarks.shape)

    if faces is not None:
        # print('find', faces.shape[0], 'faces')
        if faces.shape[0] > 1:
            print('Multi Faces detected!')
            print(img_path)
        return img, landmarks[0]

    return img, []


if __name__ == "__main__":
    retina_face('t1.jpg', 'test_detector.jpg')
