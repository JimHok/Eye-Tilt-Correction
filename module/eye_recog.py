import numpy as np


def gaborconvolve_f(img, minw_length, mult, sigma_f):
    """
    Convolve each row of an imgage with 1D log-Gabor filters.
    """
    rows, ndata = img.shape
    logGabor_f = np.zeros(ndata)
    filterb = np.zeros([rows, ndata], dtype=complex)

    radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
    radius[0] = 1

    # filter wavelength
    wavelength = minw_length

    # radial filter component
    fo = 1 / wavelength
    logGabor_f[0: int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) /
                                             (2 * np.log(sigma_f)**2))
    logGabor_f[0] = 0

    # convolution for each row
    # Not optimized version
    # for r in range(rows):
    #     signal = img[r, 0:ndata]
    #     imagefft = np.fft.fft(signal)
    #     filterb[r, :] = np.fft.ifft(imagefft * logGabor_f)

    # Optimized version
    signals = img[:, 0:ndata]
    imagefft = np.fft.fft(signals, axis=1)
    filterb = np.fft.ifft(imagefft * logGabor_f, axis=1)

    return filterb


def encode_eye(arr_polar, minw_length=18, mult=1, sigma_f=0.5):
    """
    Generate eye template and noise mask from the normalised iris region.
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

    return template
