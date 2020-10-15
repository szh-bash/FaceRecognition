import os
import cv2
import sys
import math
import time
import numpy as np
from PIL import Image
from mtcnn import MTCNN
# import face_recognition
import progressbar as pb
from multiprocessing import Process, Lock, Value
# from collections import defaultdict
sys.path.append('..')
from config import dataPath


point_96 = [[30.2946, 51.6963],  # 112x96的目标点
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655]]
point_112 = [[30.2946+8.0000, 51.6963],  # 112x112的目标点
               [65.5318+8.0000, 51.6963],
               [48.0252+8.0000, 71.7366],
               [33.5493+8.0000, 92.3655],
               [62.7299+8.0000, 92.3655]]


def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle


def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def corp_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    pil_img = Image.fromarray(image_array)
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def warp_im(img_im, orgi_landmarks, tar_landmarks):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst


def key_point(landmarks):
    left_eye = np.array(landmarks['left_eye'])
    left_eye = left_eye.sum(axis=0) / left_eye.shape[0]
    right_eye = np.array(landmarks['right_eye'])
    right_eye = right_eye.sum(axis=0) / right_eye.shape[0]
    nose_tip = np.array(landmarks['nose_tip'])
    nose_tip = nose_tip.sum(axis=0) / nose_tip.shape[0]
    left_lip0 = np.array(landmarks['top_lip'])[np.array(landmarks['top_lip']).argmin(axis=0)[0]]
    left_lip1 = np.array(landmarks['bottom_lip'])[np.array(landmarks['bottom_lip']).argmin(axis=0)[0]]
    right_lip0 = np.array(landmarks['top_lip'])[np.array(landmarks['top_lip']).argmax(axis=0)[0]]
    right_lip1 = np.array(landmarks['bottom_lip'])[np.array(landmarks['bottom_lip']).argmax(axis=0)[0]]
    res = np.array([left_eye, right_eye, nose_tip, ((left_lip0 + left_lip1) / 2), ((right_lip0 + right_lip1) / 2)])
    return res


def deal_face(img_path):
    image = cv2.imread(img_path)
    face_landmarks_list = face_recognition.face_landmarks(image, model="large")
    if len(face_landmarks_list) == 0:
        return []
    face_landmarks_dict = face_landmarks_list[0]
    aligned_face, eye_center, angle = align_face(image_array=image, landmarks=face_landmarks_dict)
    rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                         eye_center=eye_center, angle=angle, row=image.shape[0])
    cropped_face, left, top = corp_face(image_array=aligned_face, landmarks=rotated_landmarks)
    # transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top)
    # warped_face = warp_im(cropped_face, key_point(transferred_landmarks), point_112)[:112, :112, :].copy()
    normed_face = cv2.resize(cropped_face, (112, 112))
    return normed_face


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def mtcnn_align(img_path, det):
    img = cv2.imread(img_path)
    result = det.detect_faces(img)
    if len(result) == 0:
        return []
    keypoints = result[0]["keypoints"]
    origin_landmarks = [keypoints[x] for x in keypoints]
    img = warp_im(img, origin_landmarks, point_112)[:112, :112, :].copy()
    return img


def worker(le, ri):
    global lock
    global count
    global pgb
    detector = MTCNN()
    for j in range(le, ri):
        msg = q[j]
        with lock:
            pgb.update(count.value)
            count.value += 1
        if mode == 'face_recognition':
            face = deal_face(msg[0])
        elif mode == 'mtcnn':
            face = mtcnn_align(msg[0], detector)
        else:
            print('align method 404')
            exit(-1)
        if len(face) == 0:
            with lock:
                failed.value += 1
            if md == 1:
                face = cv2.imread(msg[0])
            else:
                continue
        cv2.imwrite(msg[1], face, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    q = []
    num = 16
    lock = Lock()
    count = Value('i', 0)
    failed = Value('i', 0)
    mode = 'mtcnn'
    origin_path = dataPath['Web']
    target_path = dataPath['MTWeb112P']
    if 'lfw' in origin_path:
        md = 1
        length = 13233
    else:
        md = 0
        length = 494414

    widgets = ['Dealing: ', pb.Percentage(),
               ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
               ' ', pb.Timer(),
               ' ', pb.ETA(),
               ' ', pb.FileTransferSpeed()]
    pgb = pb.ProgressBar(widgets=widgets, maxval=length)

    print(origin_path+' to '+target_path)
    print('Align mode: %s' % mode)
    path_dir = os.listdir(origin_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for allDir in path_dir:
        child = os.path.join('%s/%s' % (origin_path, allDir))
        child_dir = os.listdir(child)
        acc = os.path.join('%s/%s' % (target_path, allDir))
        if not os.path.exists(acc):
            os.mkdir(acc)
        for allSon in child_dir:
            son = os.path.join('%s/%s' % (child, allSon))
            acs = os.path.join('%s/%s' % (acc, allSon))
            q.append([son, acs])
    print('DATA LOADED!')
    print(np.array(q).shape)
    pgb.start()

    p = []
    for i in range(num-1):
        p.append(Process(target=worker, args=(length//num*i, length//num*(i+1))))
    p.append(Process(target=worker, args=(length//num*(num-1), length)))
    for i in range(num):
        p[i].start()
    for i in range(num):
        p[i].join()
    pgb.finish()
    print('succeed: %d' % (count.value-failed.value))
    print('failed: %d' % failed.value)
    print('MISSION COMPLETED!')
