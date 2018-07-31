import utils
import argparse
import random

import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
from PIL import Image as pil_image

from train import TrainingData

import numpy as np

globals = utils.getGlobals()


def show_whale(imgs, per_row=2):
    n = len(imgs)
    rows = (n + per_row - 1) // per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
    for ax in axes.flatten():
        ax.axis('off')
    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())):
        ax.imshow(img.convert('RGB'))
    #
    # I had to add this. Maybe in jupyter the showing is automatic?
    #
    plt.show()


def show_similar_image_example(config, mappings):
    for h, ps in mappings.h2ps.items():
        if len(ps) > 2:
            print('Images:', ps)
            imgs = [pil_image.open(config.filename(p)) for p in ps]
            show_whale(imgs, per_row=len(ps))
            break


def show_images(config, p):
    imgs = [
        utils.read_raw_image(config, p),
        array_to_img(utils.read_cropped_image(globals, config, p, False)),
        array_to_img(utils.read_cropped_image(globals, config, p, True))
    ]
    show_whale(imgs, per_row=3)


def show_results(a, b):
    # First pair is for matching whale
    imgs = [array_to_img(a[0]), array_to_img(b[0])]
    show_whale(imgs, per_row=2)

    # Second pair is for different whales
    imgs = [array_to_img(a[1]), array_to_img(b[1])]
    show_whale(imgs, per_row=2)


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
args = parser.parse_args()

tagged = utils.getTrainData(args.datadir)
config = utils.getConfig(args.datadir, list(tagged.keys()))
mappings = utils.getMappings(config, tagged)

show_similar_image_example(config, mappings)
show_images(config, list(mappings.h2p.values())[31])  # Show sample image

train = utils.getTrainingHashes(mappings.w2hs)

# Test on a batch of 32 with random costs.
score = np.random.random_sample(size=(len(train), len(train)))

data = TrainingData(globals, config, mappings, train, score)
(a, b), c = data[0]
print(a.shape, b.shape, c.shape)

show_results(a, b)
