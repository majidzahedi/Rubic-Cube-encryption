import numpy as np
import cv2


def enc(src, seed, iteration=1):
    np.random.seed(seed)

    if src is None:
        raise Exception("Please provide a valid source!")

    image = src.copy()
    (M, N) = image.shape
    kc = np.random.randint(0, 255, M, np.uint8)
    kr = np.random.randint(0, 255, N, np.uint8)
    for i in range(iteration):
        a = np.sum(image, 1)
        ma = np.mod(a, 2)

        for i in range(M):
            if ma[i] == 1:
                image[i, :] = np.roll(image[i, :], -1*kr[i])
            else:
                image[i, :] = np.roll(image[i, :], kr[i])

        b = np.sum(image, 0)
        mb = np.mod(b, 2)

        for i in range(N):
            if mb[i] == 1:
                image[:, i] = np.roll(image[:, i], kc[i])
            else:
                image[:, i] = np.roll(image[:, i], -1*kc[i])

        image_scrambled = image.copy()

        flip_kr = np.flip(kr)
        flip_kc = np.flip(kc)

        i1 = np.zeros((M, N), np.uint8)

        for i in range(1, len(kr)+1):
            if np.mod(i, 2) == 1:
                i1[i-1, :] = np.bitwise_xor(image_scrambled[i-1, :], kc)
            else:
                i1[i-1, :] = np.bitwise_xor(image_scrambled[i-1, :], flip_kc)

        encrypted_image = np.zeros((M, N), np.uint8)

        for i in range(1, len(kc)+1):
            if np.mod(i, 2) == 1:
                encrypted_image[:, i-1] = np.bitwise_xor(i1[:, i-1], kr)
            else:
                encrypted_image[:, i-1] = np.bitwise_xor(i1[:, i-1], flip_kr)

        image = encrypted_image

    return encrypted_image


def rubic_enc(src, seed, iteration=1):

    copy = src.copy()
    if src is None:
        return
    elif len(src.shape) == 3:
        b, g, r = cv2.split(src)

        b = enc(b, seed, iteration)
        g = enc(g, seed, iteration)
        r = enc(r, seed, iteration)
        copy = cv2.merge((b, g, r))

        return copy
    else:
        return enc(src, seed, iteration)
