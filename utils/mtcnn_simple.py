import os
from mtcnn import MTCNN
import cv2
import numpy as np
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
detector = MTCNN()


def run(src, dst):
    image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    if len(result) == 0:
        cv2.imwrite(dst, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return -1

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    # bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    def warp_affine(image, points, scale=1.0):
        eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        dy = points[1][1] - points[0][1]
        dx = points[1][0] - points[0][0]
        # 计算旋转角度
        angle = cv2.fastAtan2(dy, dx)
        rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)
        rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))
        return rot_img

    pa = [list(keypoints['left_eye']), list(keypoints['right_eye'])]
    image = warp_affine(image, pa)
    # result = detector.detect_faces(image)
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    # bounding_box = result[0]['box']
    # keypoints = result[0]['keypoints']
    # cv2.rectangle(image,
    #               (bounding_box[0], bounding_box[1]),
    #               (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
    #               (0,155,255),
    #               2)
    # cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    cv2.imwrite(dst, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return 0


# run('/dev/shm/CASIA-WebFace/0000045/011.jpg', 'test.jpg')
