import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import math
import imutils
import pandas as pd
from tqdm.auto import trange, tqdm
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d


class IrisLoc:
    def __init__(self, path=None, img=None):
        if path is not None:
            self.img = self.read_image(path)
        else:
            self.img = img

    def read_image(self, path):
        img = cv2.imread(path)
        gray_eye_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_eye_image

    def find_pupil_new(self):
        img = cv2.medianBlur(self.img, 15)
        img = cv2.Canny(img, 0, 50)
        param1 = 200
        param2 = 120
        decrement = 1
        circles = None
        while circles is None and param2 > 20:
            circles = cv2.HoughCircles(
                img,
                cv2.HOUGH_GRADIENT,
                1,
                1,
                param1=param1,
                param2=param2,
                minRadius=20,
                maxRadius=80,
            )
            if circles is not None:
                break
            param2 -= decrement
        if circles is None:
            return None, None, None
        return circles.astype(int)[0][0]

    def lash_removal_daugman(self, thresh=40):
        ref = self.img < thresh
        coords = np.where(ref == 1)
        rmov_img = self.img.astype(float)
        rmov_img[coords] = float("nan")
        temp_img = rmov_img.copy()
        temp_img[coords] = 255 / 2
        avg = np.sum(temp_img) / (rmov_img.shape[0] * rmov_img.shape[1])
        rmov_img[coords] = avg
        noise_img = np.zeros(self.img.shape)
        noise_img[coords] = 1
        return rmov_img, noise_img.astype(bool)

    def localization(self, N=400, alpha=1.6, beta=500, gamma=0.05):
        DoG = cv2.GaussianBlur(self.img, (3, 3), 0) - cv2.GaussianBlur(
            self.img, (25, 25), 0
        )
        median1 = cv2.medianBlur(DoG, 9)
        eroted = cv2.erode(median1, np.ones((3, 3), np.uint8), iterations=1)
        median2 = cv2.medianBlur(eroted, 5)
        dilated = cv2.dilate(median2, np.ones((3, 3), np.uint8), iterations=1)
        eroted = cv2.erode(dilated, np.ones((5, 5), np.uint8), iterations=1)
        result = cv2.bitwise_or(self.img, eroted)
        x, y, rad = self.find_pupil_new()
        if x is None:
            x, y = 350, 250
        s = np.linspace(0, 2 * np.pi, 400)
        c = x + 150 * np.cos(s)
        r = y + 150 * np.sin(s)
        init = np.array([r, c]).T
        snake = active_contour(result, init, alpha=alpha, beta=beta, gamma=gamma)
        return init, snake, (x, y, rad)

    def localization_seg(self, circle, alpha=1.6, beta=500, gamma=0.05):
        x, y, rad = circle
        s = np.linspace(0, 2 * np.pi, 400)
        c = x + 150 * np.cos(s)
        r = y + 150 * np.sin(s)
        init = np.array([r, c]).T
        snake = active_contour(self.img, init, alpha=alpha, beta=beta, gamma=gamma)
        return init, snake, (x, y, rad)


class IrisNorm(IrisLoc):
    def __init__(self, path, img, M=64, N=400, offset=0):
        super().__init__(path, img)
        self.M = M
        self.N = N
        self.offset = offset

    def trans_axis(self, circle, theta):
        x0, y0, r = circle
        x = int(x0 + r * math.cos(theta))
        y = int(y0 + r * math.sin(theta))
        return x, y

    def normalization(self):
        return self._normalization(self.img, self.pupil_circle, self.iris_circle)

    def normalization_eye(self):
        return self._normalization_eye(self.img, self.pupil_circle, self.iris_circle)

    def normalization_seg(self):
        return self._normalization_seg(self.img, self.pupil_circle, self.iris_circle)

    def interpolate_pixel(self, contour):
        interp_x = interp1d(np.arange(contour.shape[0]), contour[:, 0], kind="cubic")
        interp_y = interp1d(np.arange(contour.shape[0]), contour[:, 1], kind="cubic")
        contour_inter = np.zeros((self.N, 2))
        contour_inter[:, 0] = interp_x(np.linspace(0, contour.shape[0] - 1, self.N))
        contour_inter[:, 1] = interp_y(np.linspace(0, contour.shape[0] - 1, self.N))
        max_x_idx = np.argmax(contour_inter[:, 1])
        contour_inter = np.append(
            contour_inter[max_x_idx:], contour_inter[:max_x_idx], axis=0
        )
        return contour_inter

    def plot_norm_map(self, map_area, contours=None):
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.imshow(self.img, cmap="gray")
        colors = ["red", "green", "blue", "orange", "purple"]
        for i, maps in enumerate(map_area):
            plt.plot(maps[:, 1], maps[:, 0], linewidth=2, color=colors[i % len(colors)])
        if contours is not None:
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color="yellow")
        plt.show()

    def _normalization(self, img, pupil_circle, iris_circle):
        normalized, map_area = self._normalize(
            img, pupil_circle, iris_circle, self.trans_axis
        )
        return normalized, np.array(map_area)

    def _normalization_seg(self, img, pupil_circle, iris_circle):
        normalized, map_area = self._normalize(img, pupil_circle, iris_circle)
        return normalized, np.array(list(map_area))

    def _normalization_eye(self, img, pupil_circle, iris_circle):
        normalized, map_area = self._normalize(
            img, pupil_circle, iris_circle, self.trans_axis, self.trans_axis
        )
        return normalized, np.array(map_area)

    def _normalize(
        self, img, pupil_circle, iris_circle, begin_trans=None, end_trans=None
    ):
        normalized = np.zeros((self.M, self.N))
        theta = np.linspace(0, 2 * np.pi, self.N)
        map_area = []
        for i in range(self.N):
            curr_theta = theta[i] + self.offset
            if curr_theta > 2 * np.pi:
                curr_theta -= 2 * np.pi
            begin = (
                begin_trans(pupil_circle, curr_theta) if begin_trans else pupil_circle
            )
            end = end_trans(iris_circle, curr_theta) if end_trans else iris_circle
            xspace = np.linspace(begin[0], end[0], self.M)
            yspace = np.linspace(begin[1], end[1], self.M)
            normalized[:, i] = [
                img[int(y), int(x)]
                if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                else 0
                for x, y in zip(xspace, yspace)
            ]
            map_area.append(
                [
                    [int(y), int(x)]
                    if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                    else 0
                    for x, y in zip(xspace, yspace)
                ]
            )
        return normalized, map_area


class IrisMatch(IrisLoc):
    def __init__(self, path, img, snake, circles, minw_length, mult, sigma_f):
        super().__init__(path, img)
        self.snake = snake
        self.circles = circles
        self.minw_length = minw_length
        self.mult = mult
        self.sigma_f = sigma_f

    def masked(self):
        mask1 = np.zeros_like(self.img)
        mask1 = cv2.circle(
            mask1,
            (int(self.circles[0]), int(self.circles[1])),
            int(self.circles[2]),
            (255, 255, 255),
            -1,
        )
        mask2 = np.zeros_like(self.img)
        mask2[self.snake[:, 0].astype(int), self.snake[:, 1].astype(int)] = 255

        contours, _ = cv2.findContours(mask2, 2, 2)
        for i in range(len(contours)):
            cv2.drawContours(mask2, contours, i, (255, 255, 255), 3, cv2.LINE_8)

        contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            cv2.drawContours(mask2, [cnt], -1, 255, -1)

        mask = cv2.subtract(mask2, mask1)
        masked_gray = cv2.bitwise_and(self.img, self.img, mask=mask)
        return masked_gray

    def gaborconvolve_f(self, img):
        rows, ndata = img.shape
        logGabor_f = np.zeros(ndata)
        filterb = np.zeros([rows, ndata], dtype=complex)

        radius = np.arange(ndata / 2 + 1) / (ndata / 2) / 2
        radius[0] = 1

        wavelength = self.minw_length
        fo = 1 / wavelength
        logGabor_f[0 : int(ndata / 2) + 1] = np.exp(
            (-((np.log(radius / fo)) ** 2)) / (2 * np.log(self.sigma_f) ** 2)
        )
        logGabor_f[0] = 0

        signals = img[:, 0:ndata]
        imagefft = np.fft.fft(signals, axis=1)
        filterb = np.fft.ifft(imagefft * logGabor_f, axis=1)

        return filterb

    def encode_iris(self, arr_polar, arr_noise):
        filterb = self.gaborconvolve_f(arr_polar)
        l = arr_polar.shape[1]
        template = np.zeros([arr_polar.shape[0], 2 * l])
        h = np.arange(arr_polar.shape[0])

        mask_noise = np.zeros(template.shape)
        filt = filterb[:, :]

        H1 = np.real(filt) > 0
        H2 = np.imag(filt) > 0
        H3 = np.abs(filt) < 0.0001
        for i in range(l):
            ja = 2 * i

            template[:, ja] = H1[:, i]
            template[:, ja + 1] = H2[:, i]
            mask_noise[:, ja] = arr_noise[:, i] | H3[:, i]
            mask_noise[:, ja + 1] = arr_noise[:, i] | H3[:, i]

        return template, mask_noise

    def shiftbits_ham(self, template, noshifts):
        templatenew = np.zeros(template.shape)
        width = template.shape[1]
        s = 2 * np.abs(noshifts)
        p = width - s

        if noshifts == 0:
            templatenew = template

        elif noshifts < 0:
            x = np.arange(p)
            templatenew[:, x] = template[:, s + x]
            x = np.arange(p, width)
            templatenew[:, x] = template[:, x - p]

        else:
            x = np.arange(s, width)
            templatenew[:, x] = template[:, x - s]
            x = np.arange(s)
            templatenew[:, x] = template[:, p + x]

        return templatenew

    def HammingDistance(self, template1, mask1, template2, mask2):
        hd = np.nan
        shift = 0

        for shifts in range(-250, 250):
            template1s = self.shiftbits_ham(template1, shifts)
            mask1s = self.shiftbits_ham(mask1, shifts)

            mask = np.logical_and(mask1s, mask2)
            nummaskbits = np.sum(mask == 1)
            totalbits = template1s.size - nummaskbits

            C = np.logical_xor(template1s, template2)
            C = np.logical_and(C, np.logical_not(mask))
            bitsdiff = np.sum(C == 1)

            if totalbits == 0:
                hd = np.nan
            else:
                hd1 = bitsdiff / totalbits
                if hd1 < hd or np.isnan(hd):
                    hd = hd1
                    shift = shifts

        return hd, shift


def read_image(path):
    img = cv2.imread(path)
    gray_eye_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_eye_image


def find_pupil_new(img):
    img = cv2.medianBlur(img, 15)
    img = cv2.Canny(img, 0, 50)
    param1 = 200  # 200
    param2 = 120  # 150
    decrement = 1
    circles = None
    while circles is None and param2 > 20:
        # HoughCircles
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            1,
            param1=param1,
            param2=param2,
            minRadius=20,
            maxRadius=80,
        )

        if circles is not None:
            break

        param2 -= decrement

    if circles is None:
        return None, None, None

    return circles.astype(int)[0][0]


def find_pupil(img):
    param1 = 200  # 200
    param2 = 40  # 150
    decrement = 1
    circles = None

    # HoughCircles
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        1,
        1,
        param1=param1,
        param2=param2,
        minRadius=20,
        maxRadius=80,
    )

    param2 -= decrement

    if circles is None:
        return None, None, None

    return circles.astype(int)[0][0]


def lash_removal_daugman(img, thresh=40):
    ref = img < thresh
    coords = np.where(ref == 1)
    rmov_img = img.astype(float)
    rmov_img[coords] = float("nan")
    temp_img = rmov_img.copy()
    temp_img[coords] = 255 / 2
    avg = np.sum(temp_img) / (rmov_img.shape[0] * rmov_img.shape[1])
    rmov_img[coords] = avg

    noise_img = np.zeros(img.shape)
    noise_img[coords] = 1
    return rmov_img, noise_img.astype(bool)


def localization(img, N=400, alpha=1.6, beta=500, gamma=0.05):
    DoG = cv2.GaussianBlur(img, (3, 3), 0) - cv2.GaussianBlur(img, (25, 25), 0)
    median1 = cv2.medianBlur(DoG, 9)
    eroted = cv2.erode(median1, np.ones((3, 3), np.uint8), iterations=1)
    median2 = cv2.medianBlur(eroted, 5)
    dilated = cv2.dilate(median2, np.ones((3, 3), np.uint8), iterations=1)
    eroted = cv2.erode(dilated, np.ones((5, 5), np.uint8), iterations=1)
    result = cv2.bitwise_or(img, eroted)

    x, y, rad = find_pupil_new(img)

    if x is None:
        x, y = 350, 250

    s = np.linspace(0, 2 * np.pi, 400)
    c = x + 150 * np.cos(s)
    r = y + 150 * np.sin(s)
    init = np.array([r, c]).T

    snake = active_contour(result, init, alpha=alpha, beta=beta, gamma=gamma)

    return init, snake, (x, y, rad)


def localization_seg(img, circle, alpha=1.6, beta=500, gamma=0.05):
    x, y, rad = circle

    s = np.linspace(0, 2 * np.pi, 400)
    c = x + 150 * np.cos(s)
    r = y + 150 * np.sin(s)
    init = np.array([r, c]).T

    snake = active_contour(img, init, alpha=alpha, beta=beta, gamma=gamma)

    return init, snake, (x, y, rad)


def trans_axis(circle, theta):
    x0, y0, r = circle
    x = int(x0 + r * math.cos(theta))
    y = int(y0 + r * math.sin(theta))
    return x, y


def normalization(img, pupil_circle, iris_circle, M=64, N=400, offset=0):
    normalized = np.zeros((M, N))
    theta = np.linspace(0, 2 * np.pi, N)
    map_area = []

    for i in range(N):
        curr_theta = theta[i] + offset
        if curr_theta > 2 * np.pi:
            curr_theta -= 2 * np.pi
        begin = trans_axis(pupil_circle, curr_theta)
        end = iris_circle

        xspace = np.linspace(begin[0], end[i][0], M)
        yspace = np.linspace(begin[1], end[i][1], M)
        normalized[:, i] = [
            img[int(y), int(x)]
            if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
            else 0
            for x, y in zip(xspace, yspace)
        ]
        map_area.append(
            [
                [int(y), int(x)]
                if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                else 0
                for x, y in zip(xspace, yspace)
            ]
        )

    return normalized, np.array(map_area)


def normalization_eye(img, pupil_circle, iris_circle, M=64, N=400, offset=0):
    normalized = np.zeros((M, N))
    theta = np.linspace(0, 2 * np.pi, N)
    map_area = []

    for i in range(N):
        curr_theta = theta[i] + offset
        if curr_theta > 2 * np.pi:
            curr_theta -= 2 * np.pi
        begin = trans_axis(pupil_circle, curr_theta)
        end = trans_axis(iris_circle, curr_theta)

        xspace = np.linspace(begin[0], end[0], M)
        yspace = np.linspace(begin[1], end[1], M)
        normalized[:, i] = [
            img[int(y), int(x)]
            if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
            else 0
            for x, y in zip(xspace, yspace)
        ]
        map_area.append(
            [
                [int(y), int(x)]
                if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                else 0
                for x, y in zip(xspace, yspace)
            ]
        )

    return normalized, map_area


def normalization_seg(img, pupil_circle, iris_circle, M=64, N=400, offset=0):
    normalized = np.zeros((M, N))
    map_area = []

    for i in range(N):
        begin = pupil_circle
        end = iris_circle

        xspace = np.linspace(begin[i][0], end[i][0], M)
        yspace = np.linspace(begin[i][1], end[i][1], M)
        normalized[:, i] = [
            img[int(x), int(y)]
            if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
            else 0
            for x, y in zip(xspace, yspace)
        ]
        map_area.append(
            list(
                [
                    [int(y), int(x)]
                    if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                    else 0
                    for x, y in zip(xspace, yspace)
                ]
            )
        )

    return normalized, np.array(list(map_area))


def interpolate_pixel(contour, N):
    # Create a function to interpolate the x and y coordinates separately
    interp_x = interp1d(np.arange(contour.shape[0]), contour[:, 0], kind="cubic")
    interp_y = interp1d(np.arange(contour.shape[0]), contour[:, 1], kind="cubic")

    # Create a new array with N evenly spaced points
    contour_inter = np.zeros((N, 2))
    contour_inter[:, 0] = interp_x(np.linspace(0, contour.shape[0] - 1, N))
    contour_inter[:, 1] = interp_y(np.linspace(0, contour.shape[0] - 1, N))

    # Rearrange the contour_inter array so that the pixel with the greatest x-axis value is the first element
    max_x_idx = np.argmax(contour_inter[:, 1])
    contour_inter = np.append(
        contour_inter[max_x_idx:], contour_inter[:max_x_idx], axis=0
    )

    return contour_inter


def plot_norm_map(img, map_area, contours=None):
    fig, ax = plt.subplots(figsize=(20, 10))
    # Plot the image
    plt.imshow(img, cmap="gray")

    colors = ["red", "green", "blue", "orange", "purple"]
    # Plot the eye contours
    for i, maps in enumerate(map_area):
        plt.plot(maps[:, 1], maps[:, 0], linewidth=2, color=colors[i % len(colors)])

    if contours is not None:
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color="yellow")

    # Show the plot
    plt.show()


def masked(img, snake, circles):
    mask1 = np.zeros_like(img)
    mask1 = cv2.circle(
        mask1, (int(circles[0]), int(circles[1])), int(circles[2]), (255, 255, 255), -1
    )
    mask2 = np.zeros_like(img)
    mask2[snake[:, 0].astype(int), snake[:, 1].astype(int)] = 255

    contours, _ = cv2.findContours(mask2, 2, 2)
    for i in range(len(contours)):
        cv2.drawContours(mask2, contours, i, (255, 255, 255), 3, cv2.LINE_8)

    contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(mask2, [cnt], -1, 255, -1)

    mask = cv2.subtract(mask2, mask1)
    masked_gray = cv2.bitwise_and(img, img, mask=mask)
    return masked_gray


def gaborconvolve_f(img, minw_length, mult, sigma_f):
    """
    Convolve each row of an imgage with 1D log-Gabor filters.
    """
    rows, ndata = img.shape
    logGabor_f = np.zeros(ndata)
    filterb = np.zeros([rows, ndata], dtype=complex)

    radius = np.arange(ndata / 2 + 1) / (ndata / 2) / 2
    radius[0] = 1

    # filter wavelength
    wavelength = minw_length

    # radial filter component
    fo = 1 / wavelength
    logGabor_f[0 : int(ndata / 2) + 1] = np.exp(
        (-((np.log(radius / fo)) ** 2)) / (2 * np.log(sigma_f) ** 2)
    )
    logGabor_f[0] = 0

    # Optimized version
    signals = img[:, 0:ndata]
    imagefft = np.fft.fft(signals, axis=1)
    filterb = np.fft.ifft(imagefft * logGabor_f, axis=1)

    return filterb


def encode_iris(arr_polar, arr_noise, minw_length, mult, sigma_f):
    """
    Generate iris template and noise mask from the normalised iris region.
    """
    # convolve with gabor filters
    filterb = gaborconvolve_f(arr_polar, minw_length, mult, sigma_f)
    l = arr_polar.shape[1]
    template = np.zeros([arr_polar.shape[0], 2 * l])
    h = np.arange(arr_polar.shape[0])

    # making the iris template
    mask_noise = np.zeros(template.shape)
    filt = filterb[:, :]

    # quantization and check to see if the phase data is useful
    H1 = np.real(filt) > 0
    H2 = np.imag(filt) > 0

    H3 = np.abs(filt) < 0.0001
    for i in range(l):
        ja = 2 * i

        # biometric template
        template[:, ja] = H1[:, i]
        template[:, ja + 1] = H2[:, i]
        # noise mask_noise
        mask_noise[:, ja] = arr_noise[:, i] | H3[:, i]
        mask_noise[:, ja + 1] = arr_noise[:, i] | H3[:, i]

    return template, mask_noise


def shiftbits_ham(template, noshifts):
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        templatenew[:, x] = template[:, p + x]

    return templatenew


def HammingDistance(template1, mask1, template2, mask2):
    hd = np.nan
    shift = 0

    # Shifting template left and right, use the lowest Hamming distance
    for shifts in range(-250, 250):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_and(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits = template1s.size - nummaskbits

        C = np.logical_xor(template1s, template2)
        C = np.logical_and(C, np.logical_not(mask))
        bitsdiff = np.sum(C == 1)

        if totalbits == 0:
            hd = np.nan
        else:
            hd1 = bitsdiff / totalbits
            if hd1 < hd or np.isnan(hd):
                hd = hd1
                shift = shifts

    return hd, shift
