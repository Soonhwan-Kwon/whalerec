import utils

import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
from PIL import Image as pil_image


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


def show_similar_image_example(datadir, h2ps):
    for h, ps in h2ps.items():
        if len(ps) > 2:
            print('Images:', ps)
            imgs = [pil_image.open(utils.expand_path(datadir, p)) for p in ps]
            show_whale(imgs, per_row=len(ps))
            break


def show_images(globals, p):
    imgs = [
        utils.read_raw_image(globals.datadir, globals.rotate, p),
        array_to_img(utils.read_cropped_image(globals, p, False)),
        array_to_img(utils.read_cropped_image(globals, p, True))
    ]
    show_whale(imgs, per_row=3)
