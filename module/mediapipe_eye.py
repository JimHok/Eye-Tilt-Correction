import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt


def img_read(file_loc, img_num, img_take, img_side=None):
    img_l = cv2.imread(
        f'{file_loc}{str(img_num).zfill(3)}/L/S5{str(img_num).zfill(3)}L{str(img_take).zfill(2)}.jpg')
    img_r = cv2.imread(
        f'{file_loc}{str(img_num).zfill(3)}/R/S5{str(img_num).zfill(3)}R{str(img_take).zfill(2)}.jpg')
    if img_side == 'L':
        return img_l
    if img_side == 'R':
        return img_r
    img = np.concatenate((img_l, img_r), axis=1)
    return img


def detect_eye(image):
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
                249, 263, 466, 388, 387, 386, 385, 384, 398]
    # right eyes indices
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
                 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    mp_face_mesh = mp.solutions.face_mesh
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(
                int) for p in results.multi_face_landmarks[0].landmark])
            # Detect left eye
            left_eye_points = mesh_points[LEFT_EYE]
            left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
            # cv2.drawContours(image, [left_eye_points], 0, (0, 255, 0), 1)
            # cv2.circle(image, tuple(left_eye_center), 2, (0, 255, 0), -1)
            # Detect right eye
            right_eye_points = mesh_points[RIGHT_EYE]
            right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
            # cv2.drawContours(image, [right_eye_points], 0, (0, 255, 0), 1)
            # cv2.circle(image, tuple(right_eye_center), 2, (0, 255, 0), -1)
            # Detect iris
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(
                mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(
                mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            # cv2.circle(image, center_left, int(l_radius),
            #           (255, 0, 255), 1, cv2.LINE_AA)
            # cv2.circle(image, center_right, int(r_radius),
            #           (255, 0, 255), 1, cv2.LINE_AA)

        else:
            return None
    return [left_eye_points, right_eye_points, center_left, l_radius, center_right, r_radius]


def draw_eye(img, eye):
    if eye is not None:
        left_eye_points = eye[0]
        right_eye_points = eye[1]
        center_left = eye[2]
        l_radius = eye[3]
        center_right = eye[4]
        r_radius = eye[5]
        left_eye = cv2.drawContours(img, [left_eye_points], 0, (0, 255, 0), 1)
        right_eye = cv2.drawContours(
            img, [right_eye_points], 0, (0, 255, 0), 1)
        left_iris = cv2.circle(img, center_left, int(l_radius),
                               (255, 0, 255), 1, cv2.LINE_AA)
        right_iris = cv2.circle(img, center_right, int(r_radius),
                                (255, 0, 255), 1, cv2.LINE_AA)
        # Plot the image
        fig = plt.figure(figsize=(20, 10))
        plt.imshow(img, cmap='bgr')
        plt.scatter(left_eye_points[:, 0],
                    left_eye_points[:, 1], zorder=2, s=10)
        plt.scatter(right_eye_points[:, 0],
                    right_eye_points[:, 1], zorder=2, s=10)
        plt.Circle(center_left, int(l_radius), color='r', fill=False)
        plt.Circle(center_right, int(r_radius), color='r', fill=False)
        plt.show()
