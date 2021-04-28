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
    # img = cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=[0, 0, 0])
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

    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(img,
                                       thresh,
                                       scales=scales,
                                       do_flip=flip)

    if faces is not None:
        return img, landmarks[0], faces.shape[0]

    return img, [], faces.shape[0]


if __name__ == "__main__":
    retina_face('t1.jpg', 'test_detector.jpg')
