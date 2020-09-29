import os
import cv2
import sys
import math
import time
import numpy as np
from PIL import Image
import face_recognition
import progressbar as pb
from multiprocessing import Pool, Queue, Process, Lock
# from threading import Lock
from collections import defaultdict
sys.path.append('..')
from config import webPath, ACWebPath


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
    return cropped_face


q = Queue()
lock = Lock()
count = 0
widgets = ['Dealing: ', pb.Percentage(),
           ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
           ' ', pb.Timer(),
           ' ', pb.ETA(),
           ' ', pb.FileTransferSpeed()]
pgb = pb.ProgressBar(widgets=widgets, maxval=494414)


def worker():
    global q
    global lock
    global count
    global pgb
    while not q.empty():
        msg = q.get(False)
        lock.acquire()
        pgb.update(count)
        count += 1
        lock.release()
        face = deal_face(msg[0])
        if len(face) == 0:
            continue
        cv2.imwrite(msg[1], face)
    # cv2.imwrite(acs, deal_face(son))
    # global lock
    # lock.acquire()
    # global count
    # global pgb
    # pgb.update(count)
    # count += 1
    # lock.release()


if __name__ == '__main__':
    # img_path = '/dev/shm/CASIA-WebFace/6573530/054.jpg'
    # res = deal_face(img_path)
    # print(res.shape)
    # exit(0)

    num = 12

    path_dir = os.listdir(webPath)
    if not os.path.exists(ACWebPath):
        os.mkdir(ACWebPath)
    for allDir in path_dir:
        child = os.path.join('%s/%s' % (webPath, allDir))
        child_dir = os.listdir(child)
        acc = os.path.join('Multi-%s/%s' % (ACWebPath, allDir))
        if not os.path.exists(acc):
            os.mkdir(acc)
        for allSon in child_dir:
            son = os.path.join('%s/%s' % (child, allSon))
            acs = os.path.join('%s/%s' % (acc, allSon))
            q.put([son, acs])
            # p.apply_async(worker, args=(son, acs))
    print('DATA LOADED!')
    print(q.qsize())
    pgb.start()

    p = Pool(num)
    for i in range(num):
        p.apply_async(worker)
    p.close()
    # print('There''re %d image which can''t be recognized!' % fail)
    p.join()
    pgb.finish()
    print('MISSION COMPLETED!')
