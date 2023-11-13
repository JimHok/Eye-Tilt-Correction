
import mediapipe as mp
import numpy as np
import cv2
mp_face_mesh = mp.solutions.face_mesh

# left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]

# irises Indices list
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if results.multi_face_landmarks:
            # print((results.multi_face_landmarks[0]))

            # [print(p.x, p.y, p.z ) for p in results.multi_face_landmarks[0].landmark]

            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            # cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)
            # cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(
                mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(
                mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_left, int(l_radius),
                       (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(
                r_radius), (0, 255, 0), 2, cv2.LINE_AA)

            # cv2.circle(frame, center_left, 1, (0,255,0), -1, cv2.LINE_AA)
            # cv2.circle(frame, center_right, 1, (0,255,0), -1, cv2.LINE_AA)

            # drawing on the mask
            cv2.circle(mask, center_left, int(l_radius),
                       (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(mask, center_right, int(r_radius),
                       (255, 255, 255), -1, cv2.LINE_AA)

        cv2.imshow('Mask', mask)
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
