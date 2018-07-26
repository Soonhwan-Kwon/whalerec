import sys
import platform
import utils
import debug
import argparse

from pandas import read_csv
from PIL import Image as pil_image

#
#  Switch to notebook version of tqdm if using jupyter
#
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from imagehash import phash

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
args = parser.parse_args()


def unique_hashes(p2h):
    """
    Find all images associated with a given phash value.
    """
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps:
            h2ps[h] = []
        if p not in h2ps[h]:
            h2ps[h].append(p)
    return h2ps


class Globals(object):
    pass


globals = Globals()
globals.datadir = args.datadir

# 1 =================================================

csvFile = args.datadir + "/train_test.csv" if args.test else args.datadir + "/train.csv"
tagged = dict([(p, w) for _, p, w in read_csv(csvFile).to_records()])

csvFile = args.datadir + "/sample_submission_test.csv" if args.test else args.datadir + "/sample_submission.csv"
submit = [p for _, p, _ in read_csv(csvFile).to_records()]

join = list(tagged.keys()) + submit

# print len(tagged), len(submit), len(join), list(tagged.items())[:5], submit[:5]

# 2 =================================================

globals.p2size = utils.deserialize('p2size.pickle')
if globals.p2size is None:
    globals.p2size = {}

    for p in tqdm(join):
        size = pil_image.open(utils.expand_path(globals.datadir, p)).size
        globals.p2size[p] = size

    utils.serialize(globals.p2size, 'p2size.pickle')
# print len(globals.p2size), list(globals.p2size.items())[:5]


# 3 ==================================================

globals.p2h = utils.deserialize('p2h.pickle')
if globals.p2h is None:
    # Compute phash for each image in the training and test set.
    globals.p2h = {}
    for p in tqdm(join):
        img = pil_image.open(utils.expand_path(args.datadir, p))
        h = phash(img)
        globals.p2h[p] = h

    h2ps = unique_hashes(globals.p2h)

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
    for p, h in globals.p2h.items():
        h = str(h)
        if h in h2h:
            h = h2h[h]
        globals.p2h[p] = h

    utils.serialize(globals.p2h, 'p2h.pickle')

# print len(globals.p2h), list(globals.p2h.items())[:5]

# 4 =======================================================

globals.h2ps = unique_hashes(globals.p2h)

# Notice how 25460 images use only 20913 distinct image ids.
# print len(globals.h2ps), list(globals.h2ps.items())[:5]

# 5 =======================================================

if args.debug:
    debug.show_similar_image_example(globals.datadir, globals.h2ps)

# 6 =======================================================

globals.h2p = {}
for h, ps in globals.h2ps.items():
    globals.h2p[h] = utils.prefer(ps, globals.p2size)

print len(globals.h2p), list(globals.h2p.items())[:5]

# =========================================================

#
# Just going to set this to an empty array. Martin determined which should be rotated manually by adding
# to the list as he found them. Going to just ignore these for now.
#
globals.rotate = []
globals.p2bb = None

if args.debug:
    debug.show_images(globals, list(tagged.keys())[31])  # Show sample image
