import numpy as np
import cv2


def dec(src, seed, iteration=1):

    encrypted_image = src
    np.random.seed(seed)

    if src is None:
        raise Exception('Please provide a valid source!')

    (M, N) = encrypted_image.shape

    kc = np.random.randint(0, 255, M, np.uint8)
    kr = np.random.randint(0, 255, N, np.uint8)

    flip_kr = np.flip(kr)
    flip_kc = np.flip(kc)
    for i in range(iteration):
        i2 = np.zeros((M, N), np.uint8)

        for i in range(1, len(kc)+1):
            if np.mod(i, 2) == 1:
                i2[:, i-1] = np.bitwise_xor(encrypted_image[:, i-1], kr)
            else:
                i2[:, i-1] = np.bitwise_xor(encrypted_image[:, i-1], flip_kr)

        image_scrambled_2 = np.zeros((M, N), np.uint8)

        for i in range(1, len(kr)+1):
            if np.mod(i, 2) == 1:
                image_scrambled_2[i-1, :] = np.bitwise_xor(i2[i-1, :], kc)
            else:
                image_scrambled_2[i-1, :] = np.bitwise_xor(i2[i-1, :], flip_kc)

        decrypted_image = image_scrambled_2.copy()
        Bscr = np.sum(image_scrambled_2, 0)
        Mbscr = np.mod(Bscr, 2)

        for i in range(N):
            if Mbscr[i] == 1:
                decrypted_image[:, i] = np.roll(
                    decrypted_image[:, i], -1*kc[i])
            else:
                decrypted_image[:, i] = np.roll(decrypted_image[:, i], kc[i])

        Ascr = np.sum(decrypted_image, 1)
        Mascr = np.mod(Ascr, 2)

        for i in range(M):
            if Mascr[i] == 1:
                decrypted_image[i, :] = np.roll(decrypted_image[i, :], kr[i])
            else:
                decrypted_image[i, :] = np.roll(
                    decrypted_image[i, :], -1*kr[i])

        encrypted_image = decrypted_image

    return decrypted_image


def rubic_dec(src, seed, iteration=1):

    copy = src.copy()
    if src is None:
        return
    elif len(src.shape) == 3:
        b, g, r = cv2.split(src)

        b = dec(b, seed, iteration)
        g = dec(g, seed, iteration)
        r = dec(r, seed, iteration)
        copy = cv2.merge((b, g, r))
        return copy
    else:
        return dec(src, seed, iteration)
