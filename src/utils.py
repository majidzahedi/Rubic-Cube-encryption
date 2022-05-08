import numpy as np

def diff(image, image1):

    D = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > image1[i, j]:
                D[i, j] = image[i, j] - image1[i, j]
            else:
                D[i, j] = image1[i, j] - image[i, j]

    return D

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


def uaci(image, encryption):
    D = diff(image, encryption)

    result = np.sum((D/255)) * (100 / (image.shape[0] * image.shape[1]))
    return result



def sensibility(image, iteratoin):
    npcr_score = []
    for i in range(1, iteratoin):
        key = (kr, kc, i)
        npcr_score.append(npcr(image, rubic_enc(image, key)))

    return np.average(npcr_score), np.std(npcr_score)

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

