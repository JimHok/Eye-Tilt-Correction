import numpy as np
import os
from module.img_rotate import *
from module.img_test import *
from module.densenet_seg.test import *
from skimage import measure


class ImageProcessor:
    def __init__(self, img_num, img_side, img_take, set_angle=None, expand=True):
        self.img_num = img_num
        self.img_side = img_side
        self.img_take = img_take
        self.set_angle = set_angle
        self.expand = expand
        self.img_ref = self.read_image()
        self.img_rot, self.angle = self.rotate_images()
        self.imgs = [self.img_ref, self.img_rot]
        self.pupil_circles = []
        self.templates = []
        self.masks = []

    def read_image(self):
        return Loc().read_image(
            f"image/CASIA-Iris-Thousand/{str(self.img_num).zfill(3)}/{self.img_side}/S5{str(self.img_num).zfill(3)}{self.img_side}{str(self.img_take).zfill(2)}.jpg"
        )

    def rotate_images(self):
        img_rot, angle = rotate_image(
            Image.fromarray(self.img_ref), self.set_angle, self.expand
        )
        return img_rot, angle


class TestProcessor:
    def __init__(self, img_num, img_take):
        self.img_num = img_num
        self.img_take = img_take
        self.path = "C:/Users/jimyj/Desktop/TAIST/Tokyo Tech Exchange/image/Iris_data/"
        self.img_ref = self.read_ref_img()
        self.img_rot, self.angle = self.read_rot_img()
        self.imgs = [self.img_ref, self.img_rot]
        self.pupil_circles = []
        self.templates = []
        self.masks = []

    def read_ref_img(self):
        img_names = os.listdir(f"{self.path}{str(self.img_num)}0/")
        return Loc().read_image(
            f"{self.path}{str(self.img_num)}0/{img_names[self.img_take]}"
        )

    def read_rot_img(self):
        img_names = os.listdir(f"{self.path}{str(self.img_num)}1/")
        return (
            Loc().read_image(
                f"{self.path}{str(self.img_num)}1/{img_names[self.img_take]}"
            ),
            -45,
        )


class IrisProcessor:
    def __init__(
        self,
        processor_type,
        img_num,
        img_side=None,
        img_take=None,
        set_angle=None,
        expand=True,
        plot=False,
        stlit=False,
    ):
        self.processor = None
        if processor_type == "CASIA":
            self.processor = ImageProcessor(
                img_num, img_side, img_take, set_angle, expand
            )
        elif processor_type == "Own":
            self.processor = TestProcessor(img_num, img_take)
        else:
            raise ValueError('Invalid processor_type. Must be "CASIA" or "Own".')

        self.plot = plot
        self.stlit = stlit
        self.snakes = []

    def process(self):
        for i in range(len(self.processor.imgs)):
            img = self.processor.imgs[i]

            iris_loc = Loc(img=img)
            _, snake, circles = iris_loc.localization(N=400)
            # _, snake, circles = localization(img, N=400)
            pupil_circle = circles
            self.processor.pupil_circles.append(pupil_circle)
            iris_circle = np.flip(np.array(snake).astype(int), 1)
            self.snakes.append(snake)

            if circles[2] is None:
                print(f"No Iris on: {self.img_num} {self.img_side} {self.img_take}")
                return 100, 100, 100
            else:
                iris_norm = Norm(img, pupil_circle, iris_circle)
                norm, map_area = iris_norm.normalization()
                romv_img, noise_img = iris_loc.lash_removal_daugman(norm, thresh=50)
                iris_match = Match(
                    img, snake, circles, minw_length=18, mult=1, sigma_f=0.5
                )
                template, mask_noise = iris_match.encode_iris(romv_img, noise_img)
                # iris_norm, map_area = normalization(img, pupil_circle, iris_circle)
                # romv_img, noise_img = lash_removal_daugman(iris_norm, thresh=50)
                # template, mask_noise = encode_iris(
                #     romv_img, noise_img, minw_length=18, mult=1, sigma_f=0.5
                # )
                self.processor.templates.append(template)
                self.processor.masks.append(mask_noise)

        hd_raw, shift = iris_match.HammingDistance(
            self.processor.templates[0],
            self.processor.masks[0],
            self.processor.templates[1],
            self.processor.masks[1],
        )
        # hd_raw, shift = HammingDistance(
        #     self.templates[0], self.masks[0], self.templates[1], self.masks[1]
        # )
        counter_img = Image.fromarray(self.processor.imgs[1]).rotate(360 / 400 * shift)
        counter_img = crop_image(self.processor.img_ref, counter_img)
        if self.plot:
            plot_results(
                [self.processor.img_ref, self.processor.img_rot, counter_img],
                self.processor.pupil_circles,
                self.snakes,
                self.processor.templates,
                self.processor.angle,
                round(360 / 400 * shift),
                self.stlit,
            )
        return hd_raw, self.processor.angle, shift


class EyeProcessor:
    def __init__(
        self,
        processor_type,
        img_num,
        img_side=None,
        img_take=None,
        set_angle=None,
        expand=True,
        plot=False,
        stlit=False,
    ):
        self.processor = None
        if processor_type == "CASIA":
            self.processor = ImageProcessor(
                img_num, img_side, img_take, set_angle, expand
            )
        elif processor_type == "Own":
            self.processor = TestProcessor(img_num, img_take)
        else:
            raise ValueError('Invalid processor_type. Must be "CASIA" or "Own".')

        self.plot = plot
        self.stlit = stlit
        self.eye_circles = []
        self.nroms = []
        self.model_name = "densenet"
        self.model_path = "model/densenet_seg.pkl"

    def process(self):
        for i in range(len(self.processor.imgs)):
            img = self.processor.imgs[i]
            img, img_seg = run_prediction(
                img, self.model_name, self.model_path, use_gpu=False
            )
            if len(measure.find_contours(img_seg, 0.9)) == 0:
                print(
                    f"No Eye Detected: {self.img_num} {self.img_side} {self.img_take}"
                )
                return 1, self.angle, 0
            else:
                contours_pupil = np.array(
                    [max(measure.find_contours(img_seg, 0.9), key=len)]
                )
                (y, x), rad = cv2.minEnclosingCircle(
                    np.array(contours_pupil[0], dtype=np.float32)
                )
                circle_pupil = (int(x), int(y), int(rad))
                circle_eye = (int(x), int(y), int(max(img.shape) // 2))
                pupil_circle = circle_pupil
                self.processor.pupil_circles.append(pupil_circle)
                eye_circle = circle_eye
                self.eye_circles.append(eye_circle)

                eye_loc = Loc(img=img)
                eye_norm = Norm(img, pupil_circle, eye_circle, N=800)
                norm, map_area = eye_norm.normalization_eye()
                romv_img, noise_img = eye_loc.lash_removal_daugman(norm, thresh=0)
                iris_match = Match(img, None, None, minw_length=18, mult=1, sigma_f=0.5)
                template, mask_noise = iris_match.encode_iris(romv_img, noise_img)

                # eye_norm, map_area = normalization_eye(
                #     img, pupil_circle, eye_circle, M=128, N=800
                # )
                # romv_img, noise_img = lash_removal_daugman(eye_norm, thresh=0)
                # template, mask_noise = encode_iris(
                #     romv_img, noise_img, minw_length=18, mult=1, sigma_f=0.5
                # )
                self.processor.templates.append(template)
                self.nroms.append(norm)
                self.processor.masks.append(mask_noise)
        hd_raw, shift = iris_match.HammingDistance(
            self.processor.templates[0],
            self.processor.masks[0],
            self.processor.templates[1],
            self.processor.masks[1],
        )
        counter_img = Image.fromarray(self.processor.imgs[1]).rotate(360 / 800 * shift)
        counter_img = crop_image(self.processor.img_ref, counter_img)
        if self.plot:
            plot_results(
                [self.processor.img_ref, self.processor.img_rot, counter_img],
                self.processor.pupil_circles,
                self.eye_circles,
                self.processor.templates,
                self.processor.angle,
                round(360 / 800 * shift),
                self.stlit,
            )
        return hd_raw, self.processor.angle, shift
