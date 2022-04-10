
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def enc(src, kr, kc, iteration=1, dst=None):
    # kr must be same size as rows of src
    # kc must be same size as columns of src

    if src is None:
        raise Exception("src is None")

    image = src.copy()
    # plt.imshow(image, cmap='gray')
    (M, N) = image.shape

    if (M != kr.size) & (N != kc.size):
        raise Exception('kr and kc must be same size as image shape')

    orgin_image = image.copy()

    for i in range(iteration):
        # sum of all rows --> a
        # a mod 2 --> ma
        a = np.sum(image, 1)
        ma = np.mod(a, 2)

        # shift image pixel:
        # if ma[i] == 0 --> shift pixel kr[i] times to right
        # if ma[i] == 1 --> shift pixel kr[i] times to left
        for i in range(M):
            if ma[i] == 1:
                image[i, :] = np.roll(image[i, :], -1*kr[i])
            else:
                image[i, :] = np.roll(image[i, :], kr[i])

        # sum of all columns -->b
        # b mod 2 --> mb
        b = np.sum(image, 0)
        mb = np.mod(b, 2)

        # shift image pixel:
        # if mb[i] == 0 --> shift pixel kc[i] times upward
        # if mb[i] == 1 --> shift pixel kc[i] times downward
        for i in range(N):
            if mb[i] == 1:
                image[:, i] = np.roll(image[:, i], kc[i])
            else:
                image[:, i] = np.roll(image[:, i], -1*kc[i])

        image_scrambled = image.copy()

        # filp random arrays (use in next stage)
        flip_kr = np.flip(kr)
        flip_kc = np.flip(kc)

        # create a zero array same shape as original image.
        i1 = np.zeros((M, N), np.uint8)

        # perform xor on rows of scrambled image
        # if in row with odd index --> scrambled_image[odd_rows] = (scrambled_image[odd_rows] XOR kc)
        # if in row with even index --> scrambled_image[even_rows] = (scrambled_image[even_rows] XOR fliped_kc)
        for i in range(1, len(kr)+1):
            if np.mod(i, 2) == 1:
                i1[i-1, :] = np.bitwise_xor(image_scrambled[i-1, :], kc)
            else:
                i1[i-1, :] = np.bitwise_xor(image_scrambled[i-1, :], flip_kc)

        encrypted_image = np.zeros((M, N), np.uint8)

        # perform xor on columns of scrambled image
        # if in columns with odd index --> encrypted_image[odd_columns] = (scrambled_image[odd_columns] XOR kr)
        # if in columns with even index --> encrypted_image[even_columns] = (scrambled_image[even_columns] XOR fliped_kr)
        for i in range(1, len(kc)+1):
            if np.mod(i, 2) == 1:
                encrypted_image[:, i-1] = np.bitwise_xor(i1[:, i-1], kr)
            else:
                encrypted_image[:, i-1] = np.bitwise_xor(i1[:, i-1], flip_kr)

        image = encrypted_image

    return encrypted_image


def dec(src, kr, kc, iteration=1, dst=None):
    # kr must be same size as rows of src
    # kc must be same size as columns of src

    encrypted_image = src

    # check if image loaded
    if src is None:
        raise Exception('src is None')

    (M, N) = encrypted_image.shape

    # check if the kr and kc is correct shape
    if (M != kr.size) & (N != kc.size):
        raise Exception('kr and kc must be same size as image shape')

    flip_kr = np.flip(kr)
    flip_kc = np.flip(kc)
    for i in range(iteration):
        # create a zero array same shape as original image.
        i2 = np.zeros((M, N), np.uint8)

        # perform xor on columns of encrypted image
        # if in columns with odd index --> scrambled_image[odd_columns] = (encrypted_image[odd_columns] XOR kr)
        # if in columns with even index --> scrambled_image[even_columns] = (encrypted_image[even_columns] XOR fliped_kr)
        for i in range(1, len(kc)+1):
            if np.mod(i, 2) == 1:
                i2[:, i-1] = np.bitwise_xor(encrypted_image[:, i-1], kr)
            else:
                i2[:, i-1] = np.bitwise_xor(encrypted_image[:, i-1], flip_kr)

        image_scrambled_2 = np.zeros((M, N), np.uint8)

        # perform xor on rows of scrambled image
        # if in row with odd index --> scrambled_image[odd_rows] = (scrambled_image[odd_rows] XOR kc)
        # if in row with even index --> scrambled_image[even_rows] = (scrambled_image[even_rows] XOR fliped_kc)
        for i in range(1, len(kr)+1):
            if np.mod(i, 2) == 1:
                image_scrambled_2[i-1, :] = np.bitwise_xor(i2[i-1, :], kc)
            else:
                image_scrambled_2[i-1, :] = np.bitwise_xor(i2[i-1, :], flip_kc)

        decrypted_image = image_scrambled_2.copy()
        # sum of columns
        Bscr = np.sum(image_scrambled_2, 0)
        Mbscr = np.mod(Bscr, 2)

        # shift image pixel:
        # if mb[i] == 0 --> shift pixel kc[i] times downward
        # if mb[i] == 1 --> shift pixel kc[i] times upward
        for i in range(N):
            if Mbscr[i] == 1:
                decrypted_image[:, i] = np.roll(
                    decrypted_image[:, i], -1*kc[i])
            else:
                decrypted_image[:, i] = np.roll(decrypted_image[:, i], kc[i])

        # sum of rows
        Ascr = np.sum(decrypted_image, 1)
        Mascr = np.mod(Ascr, 2)

        # shift image pixel:
        # if ma[i] == 0 --> shift pixel kr[i] times to left
        # if ma[i] == 1 --> shift pixel kr[i] times to right
        for i in range(M):
            if Mascr[i] == 1:
                decrypted_image[i, :] = np.roll(decrypted_image[i, :], kr[i])
            else:
                decrypted_image[i, :] = np.roll(
                    decrypted_image[i, :], -1*kr[i])

        encrypted_image = decrypted_image

    return decrypted_image


def rubic_enc(src, key):
    # check whether the loaded image is RGB or GRAYSCALE. if the image is RGB then performs the encryption on every
    # channel.

    kr = key[0]
    kc = key[1]
    iteration = key[2]

    copy = src.copy()
    if src is None:
        return
    elif len(src.shape) == 3:
        b, g, r = cv2.split(src)

        b = enc(b, kr, kc, iteration)
        g = enc(g, kr, kc, iteration)
        r = enc(r, kr, kc, iteration)
        copy = cv2.merge((b, g, r))

        return copy
    else:
        return enc(src, kr, kc, iteration)


def rubic_dec(src, key):
    # check whether the loaded image is RGB or GRAYSCALE. if the image is RGB then performs the decryption on every
    # channel.

    kr = key[0]
    kc = key[1]
    iteration = key[2]

    copy = src.copy()
    if src is None:
        return
    elif len(src.shape) == 3:
        b, g, r = cv2.split(src)

        b = dec(b, kr, kc, iteration)
        g = dec(g, kr, kc, iteration)
        r = dec(r, kr, kc, iteration)
        copy = cv2.merge((b, g, r))
        return copy
    else:
        return dec(src, kr, kc, iteration)


def diff(image, image1):
    # definition of diff, calculate diffrence between two image

    D = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > image1[i, j]:
                D[i, j] = image[i, j] - image1[i, j]
            else:
                D[i, j] = image1[i, j] - image[i, j]

    return D

# definition of npcr


def npcr(image, encryption):
    D = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == encryption[i, j]:
                D[i, j] = 0
            else:
                D[i, j] = 1

    result = np.sum(D) / (image.shape[0] * image.shape[1])
    return result * 100

# definition of uaci


def uaci(image, encryption):
    D = diff(image, encryption)

    result = np.sum((D/255)) * (100 / (image.shape[0] * image.shape[1]))
    return result

# calculate Key Sensibility (part 3.2.2. Key Sensibility)


def sensibility(image, iteratoin):
    npcr_score = []
    for i in range(1, iteratoin):
        key = (kr, kc, i)
        npcr_score.append(npcr(image, rubic_enc(image, key)))

    return np.average(npcr_score), np.std(npcr_score)

# calculate required time to encrypt or decrypt.


def speed_test():
    time_images = []
    for i in [64, 128, 256, 512, 1024]:
        time_images.append(np.random.randint(0, 255, (i, i)))

    def key_gen(image, l_bit=1):
        (M, N) = image.shape
        Kr = np.random.randint(0, 255, M, np.uint8)
        Kc = np.random.randint(0, 255, N, np.uint8)

        return (Kr, Kc, l_bit)

    def cal_enc_time(image):
        key = key_gen(image)
        t1 = time.perf_counter()
        rubic_enc(image, key)
        t2 = time.perf_counter()

        return t2-t1

    def cal_dec_time(image):
        key = key_gen(image)
        t1 = time.perf_counter()
        rubic_dec(image, key)
        t2 = time.perf_counter()

        return t2-t1

    dauration = [[], []]
    for image in time_images:
        dauration[0].append(cal_enc_time(image))
        dauration[1].append(cal_dec_time(image))

    date = {'Image size': ['64*64', '128*125', '256*256', '512*512', '1024*1024'],
            'Encryption time': dauration[0],
            'Decryption time': dauration[1]}
    df = pd.DataFrame(date)
    return df
