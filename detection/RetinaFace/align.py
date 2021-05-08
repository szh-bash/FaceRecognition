import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace


sys.path.append('.')


def retina_face(img_path):
    thresh = 0.8
    scales = [250, 250]

    gpuid = -1
    detector = RetinaFace('/home/shenzhonghai/FaceClustering/detection/RetinaFace/model/R50', 0, gpuid, 'net3')
    # detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if 'PIE' in img_path:
        # border 64 -> 112
        img = cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=[0, 0, 0])
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

    faces, landmarks = detector.detect_center(img,
                                              thresh,
                                              scales=scales,
                                              do_flip=flip)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if faces is not None:
        return img, landmarks

    return img, []


if __name__ == "__main__":
    retina_face('/data/shenzhonghai/lfw/lfw/Woody_Allen/Woody_Allen_0004.jpg')
