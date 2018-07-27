import sys
import platform
import argparse
import random

import utils
import debug
import model
from train import TrainingData

from imagehash import phash
from pandas import read_csv
from PIL import Image as pil_image

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
args = parser.parse_args()


class Globals(object):
    def __init__(self, datadir):
        self.datadir = datadir

    def filename(self, p):
        return utils.expand_path(self.datadir, p)


globals = Globals(args.datadir)
globals.img_shape = (384, 384, 1)  # The image shape used by the model
globals.histories = []
globals.steps = 0

#
# Just going to set this to an empty array. Martin determined which should be rotated manually by adding
# to the list as he found them. Going to just ignore these for now.
#
globals.rotate = []
#
# If we want to include bounding boxes for the images (Martin's method to obtain them was manual)
# then we can read them in here. I'm going to ignore them for now and assume that we are going to try
# and use closely cropped images.
# Similarly not setting any excluded images
#
globals.p2bb = None
globals.exclude = None

# 1 =================================================

csvFile = args.datadir + "/train_test.csv" if args.test else args.datadir + "/train.csv"
tagged = dict([(p, w) for _, p, w in read_csv(csvFile).to_records()])

csvFile = args.datadir + "/sample_submission_test.csv" if args.test else args.datadir + "/sample_submission.csv"
submit = [p for _, p, _ in read_csv(csvFile).to_records()]

join = list(tagged.keys()) + submit

# print len(tagged), len(submit), len(join), list(tagged.items())[:5], submit[:5]

# 2 =================================================

globals.p2size = utils.p2size(globals, join)

# 3 ==================================================

globals.p2h = utils.p2h(globals, join)

# 4 =======================================================

globals.h2ps = utils.unique_hashes(globals.p2h)

# Notice how 25460 images use only 20913 distinct image ids.
# print len(globals.h2ps), list(globals.h2ps.items())[:5]

# 5 =======================================================

if args.debug:
    debug.show_similar_image_example(globals)

# 6 =======================================================

globals.h2p = {}
for h, ps in globals.h2ps.items():
    globals.h2p[h] = utils.prefer(ps, globals.p2size)

# print len(globals.h2p), list(globals.h2p.items())[:5]

# 10 =========================================================

if args.debug:
    debug.show_images(globals, list(tagged.keys())[31])  # Show sample image

# 11 =========================================================

#
# THIS CAN TOTALLY BE MOVED UP THE LINE BEFORE TO JUST BEFORE THE MAKE STANDARD
#
globals.model, globals.branch_model, globals.head_model = model.build(globals.img_shape, 64e-5, 0)
# head_model.summary()
# branch_model.summary()

# 17 =========================================================

globals.h2ws = utils.h2ws(globals.p2h, tagged)

# 18 =========================================================

globals.w2hs = utils.w2hs(globals)

# 19 =========================================================

# Find the list of training images, keep only whales with at least two images.
train = []  # A list of training image ids
for hs in globals.w2hs.values():
    if len(hs) > 1:
        train += hs
random.shuffle(train)

globals.train = train
globals.train_set = set(train)

utils.map_train(globals)

if args.debug:

    # 21 =========================================================

    # Test on a batch of 32 with random costs.
    score = np.random.random_sample(size=(len(train), len(train)))
    data = TrainingData(globals, score)
    (a, b), c = data[0]
    print a.shape, b.shape, c.shape

    # 22, 23 =========================================================
    debug.show_results(a, b)

model.make_standard(globals)
