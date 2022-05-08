#!/usr/bin/python

import sys
import getopt

from src import enc
from src import dec
import cv2


def main(argv):
    inputfile = './assets/lena.png'
    outputfile = './assets/encrypted.png'
    seed = 1
    isDec = False
    try:
        opts, args = getopt.getopt(
            argv, "hi:o:s:d:", ["ifile=", "ofile=", "seed=", "Dec="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile> -s <key> -d <decrypt>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile> -s <key> -d <decrypt>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--seed"):
            seed = arg
        elif opt in ("-d", "--Dec"):
            isDec = True

    if isDec == False:
        image = cv2.imread(inputfile, 1)
        outputImage = enc.rubic_enc(image, int(seed))
        cv2.imwrite(outputfile, outputImage)
    else:
        image = cv2.imread(inputfile, 1)
        outputImage = dec.rubic_dec(image, int(seed))
        cv2.imwrite(outputfile, outputImage)


if __name__ == "__main__":
    main(sys.argv[1:])
