import sys
import platform
import utils
import debug

from os.path import isfile

from pandas import read_csv

from PIL import Image as pil_image

#
#  Switch to notebook version of tqdm if using jupyter
#
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

import pickle
from imagehash import phash
from keras.preprocessing.image import array_to_img

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
args = parser.parse_args()


def read_for_training(h2p, p2size, p2bb, rotate, p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return utils.read_cropped_image(args.datadir, h2p, p2size, p2bb, rotate, p, True)


def read_for_validation(h2p, p2size, p2bb, rotate, p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return utils.read_cropped_image(args.datadir, h2p, p2size, p2bb, rotate, p, False)


# 1 =================================================

csvFile = args.datadir + "/train_test.csv" if args.test else args.datadir + "/train.csv"
tagged = dict([(p, w) for _, p, w in read_csv(csvFile).to_records()])

csvFile = args.datadir + "/sample_submission_test.csv" if args.test else args.datadir + "/sample_submission.csv"
submit = [p for _, p, _ in read_csv(csvFile).to_records()]
join = list(tagged.keys()) + submit
# print len(tagged), len(submit), len(join), list(tagged.items())[:5], submit[:5]

# 2 =================================================

p2size = {}
for p in tqdm(join):
    size = pil_image.open(utils.expand_path(args.datadir, p)).size
    p2size[p] = size
# print len(p2size), list(p2size.items())[:5]


# 3 ==================================================

if isfile('p2h.pickle'):
    with open('p2h.pickle', 'rb') as f:
        p2h = pickle.load(f)
else:
    # Compute phash for each image in the training and test set.
    p2h = {}
    for p in tqdm(join):
        img = pil_image.open(utils.expand_path(args.datadir, p))
        h = phash(img)
        p2h[p] = h

    # Find all images associated with a given phash value.
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps:
            h2ps[h] = []
        if p not in h2ps[h]:
            h2ps[h].append(p)

    # Find all distinct phash values
    hs = list(h2ps.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    for i, h1 in enumerate(tqdm(hs)):
        for h2 in hs[:i]:
            if h1 - h2 <= 6 and utils.match(args.datadir, h2ps, h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2:
                    s1, s2 = s2, s1
                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p, h in p2h.items():
        h = str(h)
        if h in h2h:
            h = h2h[h]
        p2h[p] = h

# print len(p2h), list(p2h.items())[:5]

# 4 =======================================================

# For each image id, determine the list of pictures
for p, h in p2h.items():
    if h not in h2ps:
        h2ps[h] = []
    if p not in h2ps[h]:
        h2ps[h].append(p)
# Notice how 25460 images use only 20913 distinct image ids.
# print len(h2ps), list(h2ps.items())[:5]

# 5 =======================================================

if args.debug:
    for h, ps in h2ps.items():
        if len(ps) > 2:
            print('Images:', ps)
            imgs = [pil_image.open(utils.expand_path(args.datadir, p)) for p in ps]
            debug.show_whale(imgs, per_row=len(ps))
            break

# 6 =======================================================

h2p = {}
for h, ps in h2ps.items():
    h2p[h] = utils.prefer(ps, p2size)
# print len(h2p), list(h2p.items())[:5]

# =========================================================

#
# Just going to set this to an empty array. Martin determined which should be rotated manually by adding
# to the list as he found them. Going to just ignore these.
#
rotate = []
p2bb = None

# Show an example of a duplicate image (from training of test set)
p = list(tagged.keys())[31]
print p
imgs = [
    utils.read_raw_image(args.datadir, rotate, p),
    array_to_img(read_for_validation(h2p, p2size, p2bb, rotate, p)),
    array_to_img(read_for_training(h2p, p2size, p2bb, rotate, p))
]
print imgs
debug.show_whale(imgs, per_row=3)
