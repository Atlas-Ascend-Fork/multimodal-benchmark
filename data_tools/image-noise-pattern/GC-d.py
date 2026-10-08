from skimage import exposure


def gamma_correction(img, gamma_par=2):
    gamma_corrected = exposure.adjust_gamma(img, gamma_par)
    return gamma_corrected
