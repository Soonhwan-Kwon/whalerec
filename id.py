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


class Config(object):
    def __init__(self, datadir):
        self.datadir = datadir

    def filename(self, p):
        return utils.expand_path(self.datadir, p)


config = Config(args.datadir)
config.img_shape = (384, 384, 1)  # The image shape used by the model

#
# Just going to set this to an empty array. Martin determined which should be rotated manually by adding
# to the list as he found them. Going to just ignore these for now.
#
config.rotate = []
#
# If we want to include bounding boxes for the images (Martin's method to obtain them was manual)
# then we can read them in here. I'm going to ignore them for now and assume that we are going to try
# and use closely cropped images.
# Similarly not setting any excluded images
#
config.p2bb = None
config.exclude = None

# 1 =================================================

csvFile = args.datadir + "/train_test.csv" if args.test else args.datadir + "/train.csv"
tagged = dict([(p, w) for _, p, w in read_csv(csvFile).to_records()])

csvFile = args.datadir + "/sample_submission_test.csv" if args.test else args.datadir + "/sample_submission.csv"
submit = [p for _, p, _ in read_csv(csvFile).to_records()]

join = list(tagged.keys()) + submit

# print(len(tagged), len(submit), len(join), list(tagged.items())[:5], submit[:5])

# 2 =================================================

config.p2size = utils.p2size(config, join)

# 3 ==================================================

config.p2h = utils.p2h(config, join)

# 4 =======================================================

config.h2ps = utils.unique_hashes(config.p2h)

# Notice how 25460 images use only 20913 distinct image ids.
# print(len(config.h2ps), list(config.h2ps.items())[:5])

# 5 =======================================================

if args.debug:
    debug.show_similar_image_example(config)

# 6 =======================================================

config.h2p = {}
for h, ps in config.h2ps.items():
    config.h2p[h] = utils.prefer(ps, config.p2size)

# print(len(config.h2p), list(config.h2p.items())[:5])

# 10 =========================================================

if args.debug:
    debug.show_images(config, list(tagged.keys())[31])  # Show sample image

# 17 =========================================================

config.h2ws = utils.h2ws(config.p2h, tagged)

# 18 =========================================================

config.w2hs = utils.w2hs(config)

# 19 =========================================================


class Data(object):
    pass


data = Data()
data.histories = []
data.steps = 0

#
# THIS CAN TOTALLY BE MOVED UP THE LINE BEFORE TO JUST BEFORE THE MAKE STANDARD
#
data.model, data.branch_model, data.head_model = model.build(config.img_shape, 64e-5, 0)
# head_model.summary()
# branch_model.summary()


# Find the list of training images, keep only whales with at least two images.
train = []  # A list of training image ids
for hs in config.w2hs.values():
    if len(hs) > 1:
        train += hs
random.shuffle(train)

data.train = train
data.train_set = set(train)

utils.map_train(config, data)

if args.debug:

    # 21 =========================================================

    # Test on a batch of 32 with random costs.
    score = np.random.random_sample(size=(len(train), len(train)))
    data = TrainingData(config, train, score)
    (a, b), c = data[0]
    print(a.shape, b.shape, c.shape)

    # 22, 23 =========================================================
    debug.show_results(a, b)

model.make_standard(config, data)
