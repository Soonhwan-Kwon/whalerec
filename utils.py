from os.path import isfile
import numpy as np
# Suppress annoying stderr output when importing keras.
# old_stderr = sys.stderr
# sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
import keras
# sys.stderr = old_stderr

import random
from keras import backend as K
from keras.preprocessing.image import img_to_array
from scipy.ndimage import affine_transform

from PIL import Image as pil_image
from math import sqrt
import pickle


def p2size(globals, images):
    p2size = deserialize('p2size.pickle')
    if p2size is None:
        p2size = {}

        for imagename in tqdm(images):
            size = pil_image.open(globals.filename(imagename)).size
            p2size[imagename] = size

        serialize(p2size, 'p2size.pickle')
    # print len(p2size), list(p2size.items())[:5]
    return p2size


def p2h(globals, images):
    p2h = deserialize('p2h.pickle')
    if p2h is None:
        # Compute phash for each image in the training and test set.
        p2h = {}
        for imagename in tqdm(images):
            img = pil_image.open(globals.filename(p))
            h = phash(img)
            globals.p2h[imagename] = h

        h2ps = unique_hashes(p2h)

        # Find all distinct phash values
        hs = list(h2ps.keys())

        # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
        h2h = {}
        for i, h1 in enumerate(tqdm(hs)):
            for h2 in hs[:i]:
                if h1 - h2 <= 6 and match(globals.datadir, h2ps, h1, h2):
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

        serialize(p2h, 'p2h.pickle')

    # print len(globals.p2h), list(globals.p2h.items())[:5]
    return p2h


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


def serialize(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def deserialize(filename):
    if isfile(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return None


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


def expand_path(datadir, p):
    file = datadir + '/train/' + p
    if isfile(file):
        return file

    file = datadir + '/test/' + p

    if isfile(file):
        return file
    return p


def read_cropped_image(globals, p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed, True for training, False for validation
    @return a numpy array with the transformed image
    """
    anisotropy = 2.15  # The horizontal compression ratio

    # If an image id was given, convert to filename
    if p in globals.h2p:
        p = globals.h2p[p]
    size_x, size_y = globals.p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    if globals.p2bb is None or p not in globals.p2size.keys():
        crop_margin = 0.0
        x0 = 0
        y0 = 0
        x1 = size_x
        y1 = size_y
    else:
        crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy
        x0, y0, x1, y1 = globals.p2bb[p]
    if p in globals.rotate:
        x0, y0, x1, y1 = size_x - x1, size_y - y1, size_x - x0, size_y - y0
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if (x0 < 0):
        x0 = 0
    if (x1 > size_x):
        x1 = size_x
    if (y0 < 0):
        y0 = 0
    if (y1 > size_y):
        y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * globals.img_shape[0]], [0, 1, -0.5 * globals.img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / globals.img_shape[0], 0, 0], [0, (x1 - x0) / globals.img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(globals, p).convert('L')
    img = img_to_array(img)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(img, matrix, offset, output_shape=globals.img_shape[:-1], order=1, mode='constant', cval=np.average(img))
    img = img.reshape(globals.img_shape)

    # Normalize to zero mean and unit variance
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img


def read_raw_image(globals, p):
    img = pil_image.open(globals.filename(p))
    if p in globals.rotate:
        img = img.rotate(180)
    return img


# Two phash values are considered duplicate if, for all associated image pairs:
# 1) They have the same mode and size;
# 2) After normalizing the pixel to zero mean and variance 1.0, the mean square error does not exceed 0.1
def match(datadir, h2ps, h1, h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 = pil_image.open(expand_path(datadir, p1))
            i2 = pil_image.open(expand_path(datadir, p2))
            if i1.mode != i2.mode or i1.size != i2.size:
                return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1 / sqrt((a1**2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2 / sqrt((a2**2).mean())
            a = ((a1 - a2)**2).mean()
            if a > 0.1:
                return False
    return True


# For each images id, select the prefered image
def prefer(ps, p2size):
    if len(ps) == 1:
        return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p


def h2ws(p2h, tagged):
    h2ws = {}
    new_whale = 'new_whale'
    for p, w in tagged.items():
        if w != new_whale:  # Use only identified whales
            h = p2h[p]
            if h not in h2ws:
                h2ws[h] = []
            if w not in h2ws[h]:
                h2ws[h].append(w)
    for h, ws in h2ws.items():
        if len(ws) > 1:
            h2ws[h] = sorted(ws)
    # print len(h2ws)
    return h2ws


def w2hs(globals):
    w2hs = {}
    for h, ws in globals.h2ws.items():
        if len(ws) == 1:  # Use only unambiguous pictures
            if globals.exclude is not None and globals.h2p[h] in globals.exclude:
                print "Skipping", h  # Skip excluded images
            else:
                w = ws[0]
                if w not in w2hs:
                    w2hs[w] = []
                if h not in w2hs[w]:
                    w2hs[w].append(h)
    for w, hs in w2hs.items():
        if len(hs) > 1:
            w2hs[w] = sorted(hs)
    # print len(w2hs)
    return w2hs


def map_train(globals):
    """
    Couldn't figure out what to call this
    """

    w2ts = {}  # Associate the image ids from train to each whale id.
    for w, hs in globals.w2hs.items():
        for h in hs:
            if h in globals.train_set:
                if w not in w2ts:
                    w2ts[w] = []
                if h not in w2ts[w]:
                    w2ts[w].append(h)
    for w, ts in w2ts.items():
        w2ts[w] = np.array(ts)

    globals.w2ts = w2ts

    t2i = {}  # The position in train of each training image id
    for i, t in enumerate(globals.train):
        t2i[t] = i

    globals.t2i = t2i

    print len(globals.train), len(globals.w2ts)
