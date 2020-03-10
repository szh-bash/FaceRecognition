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


# _, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
# if points.shape[0] == 10:
#     if points.shape[1] == 1:
#         # left eye(0,5) right eye(1,6) nose(2,7) left mouth(3,8) right mouth(4,9)
#         leye = np.array((points[0], points[5])).reshape(2, )
#         reye = np.array((points[1], points[6])).reshape(2, )
#         lmouth = np.array((points[3], points[8])).reshape(2, )
#         rmouth = np.array((points[4], points[9])).reshape(2, )
#         ## 两眼中心，嘴巴中心,距离
#         ceye = (leye + reye) / 2
#         cmouth = (lmouth + rmouth) / 2
#         dis_ce_cm = np.linalg.norm(ceye - cmouth)
#         ratio = 48 / dis_ce_cm
#         ## 变换后双眼和嘴巴的x坐标
#         dis_le_re = np.linalg.norm(leye - reye)
#         dis_lm_rm = np.linalg.norm(lmouth - rmouth)
#         l_eye = np.array((144 / 2 - dis_le_re * ratio / 2, 48)).reshape(2, )
#         r_eye = np.array((144 / 2 + dis_le_re * ratio / 2, 48)).reshape(2, )
#         l_mouth = np.array((144 / 2 - dis_lm_rm * ratio / 2, 48 * 2)).reshape(2, )
#         r_mouth = np.array((144 / 2 + dis_lm_rm * ratio / 2, 48 * 2)).reshape(2, )
#         ## 透视变换，获取变换矩阵
#         src = np.array((leye, reye, lmouth, rmouth))
#         dist = np.array((l_eye, r_eye, l_mouth, r_mouth), dtype=np.float32)
#         warpMatrix = cv2.getPerspectiveTransform(src, dist)
#         img_align = cv2.warpPerspective(img, warpMatrix, (144, 144))
#         misc.imsave(output_filename, img_align)
#         print('save' + output_filename)
#         print('time cost:', time.time() - time_start)