import os
import cv2
import bz2
import dlib
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import get_file

from face3d import mesh
from face3d.morphable_model import MorphabelModel
from face3d.render_utils.render.render_operator import Render

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def reverse_to_image(vertices, h, w, is_perspective=False):
    projected_vertices = vertices.copy()
    # flip vertics along y-axis
    projected_vertices[:, 1] = h - projected_vertices[:, 1] - 1
    # move to center of image
    projected_vertices[:, 0] -= w / 2
    projected_vertices[:, 1] -= h / 2
    if is_perspective:
        projected_vertices[:, 0] /= w / 2
        projected_vertices[:, 1] /= h / 2
    return projected_vertices


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector()  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, img):
        dets = self.detector(img, 1)
        detection = dets[0]
        face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
        return detection, face_landmarks


if __name__ == '__main__':
    suffix = '_zero_expression'
    img_root = 'examples/Data/train_renamed/'
    result_root = f'examples/results/train_renamed{suffix}/'

    landmark_image_path, fitted_image_path, mean_neutral_path = list(map(lambda x: os.path.join(result_root, x),
                                                                         ['landmark', 'fitted', 'mean_neutral']))
    _ = list(map(lambda x: ensure_dir(x), [landmark_image_path, fitted_image_path, mean_neutral_path]))

    landmarks_model_path = unpack_bz2(
        get_file('shape_predictor_68_face_landmarks.dat.bz2', LANDMARKS_MODEL_URL, cache_subdir='temp'))
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    bfm = MorphabelModel('examples/Data/BFM/Out/BFM.mat')
    RenderAgent = Render()

    for img_name in tqdm(os.listdir(img_root)):
        raw_img_path = os.path.join(img_root, img_name)
        img = cv2.imread(raw_img_path)
        # landmarks detection
        detection, face_landmarks = landmarks_detector.get_landmarks(img)

        show_img = img.copy()
        show_img = cv2.rectangle(show_img, (detection.left(), detection.top()), (detection.right(), detection.bottom()),
                                 (0, 255, 0), 2)
        for i, landmark in enumerate(face_landmarks):
            cv2.circle(show_img, landmark, 2, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(landmark_image_path, f'ldmk_{img_name}'), show_img)

        h, w, _ = img.shape
        x_face_landmarks = np.array(face_landmarks).astype('float64')
        projected_vertices = reverse_to_image(x_face_landmarks, h, w)
        X_ind = bfm.kpt_ind
        # fitting
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(projected_vertices, X_ind, max_iter=10,
                                                                          withExpression=False)

        # rendering fitted parameters
        fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)

        transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
        transformed_vertices = bfm.transform(transformed_vertices, 1, [0, 180, 0], [0, 0, 0])
        image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
        fitted_image = np.flip(RenderAgent(np.flip(img, 1), image_vertices, bfm.triangles), 1).copy()
        cv2.imwrite(os.path.join(fitted_image_path, f'fitted{suffix}_{img_name}'), fitted_image)

        image_landmarks = image_vertices[bfm.kpt_ind, :2]
        for i, landmark in enumerate(image_landmarks):
            cv2.circle(fitted_image, tuple((landmark).astype('int')), 2, (0, 0, 255), -1)
        for i, landmark in enumerate(face_landmarks):
            cv2.circle(fitted_image, landmark, 2, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(fitted_image_path, f'fitted_with_ldmk{suffix}_{img_name}'), fitted_image)

        fitted_vertices = bfm.generate_vertices(fitted_sp, bfm.get_exp_para('zero'))

        transformed_vertices = bfm.transform(fitted_vertices, fitted_s, [0, 180, 0], [0, 0, 0])
        image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
        image_render = RenderAgent(np.ones_like(img), image_vertices, bfm.triangles)
        cv2.imwrite(os.path.join(mean_neutral_path, f'mean_neutral_render{suffix}_{img_name}'), image_render)

        transformed_vertices = bfm.transform(fitted_vertices, fitted_s, [0, -90, 0], [w / 4, 0, 0])
        image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
        image_render = RenderAgent(np.ones_like(img), image_vertices, bfm.triangles)
        cv2.imwrite(os.path.join(mean_neutral_path, f'mean_neutral_render{suffix}_left_{img_name}'), image_render)
