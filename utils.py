import platform
import random
import csv

import os
import numpy as np
# Suppress annoying stderr output when importing keras.
# old_stderr = sys.stderr
# sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
import keras
# sys.stderr = old_stderr

from keras import backend as K
from keras.preprocessing.image import img_to_array
from scipy.ndimage import affine_transform

from PIL import Image as pil_image
from math import sqrt
import pickle
#
#  Switch to notebook version of tqdm if using jupyter
#
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from imagehash import phash


#
# Don't put these in debug since that has matlab stuff that I don't want to import in the general case
# as it fails on headless servers.
#
def debug_var(name, var):
    if isinstance(var, list):
        print(name + ":", "size:", len(var), "sample:", var[:5])
    elif isinstance(var, dict):
        print(name + ":", "size:", len(var), "sample:", list(var.items())[:5])
    else:
        print(name + ":", var)


class Globals(object):
    img_shape = (384, 384, 1)  # The image shape used by the model


class ImageInfo(object):
    size = None
    hash = None
    #
    # Just going to set this to an empty array. Martin determined which should be rotated manually by adding
    # to the list as he found them. Going to just ignore these for now.
    #
    rotate = False
    #
    # If we want to include bounding boxes for the images (Martin's method to obtain them was manual)
    # then we can read them in here. I'm going to ignore them for now and assume that we are going to try
    # and use closely cropped images.
    #
    bb = None


class ImageSet(object):
    def __init__(self, rootdir):
        self.rootdir = rootdir
        # Maps filename to its ImageInfo
        self.infomap = {}

    def filename(self, p):
        return os.path.join(self.rootdir, p)


class Mappings(object):
    pass


def serialize(obj, setname, objname):
    directory = os.path.join("sets", setname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, objname + ".pickle")
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def deserialize(setname, name):
    filename = os.path.join("sets", setname, name + ".pickle")
    if os.path.isfile(filename):
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


def hashes2images(h2p, hashes):
    images = []
    for hash in hashes:
        if hash in h2p:
            images.append(h2p[hash])
    return images


def read_cropped_image(globals, imageset, p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed, True for training, False for validation
    @return a numpy array with the transformed image
    """
    anisotropy = 2.15  # The horizontal compression ratio

    info = imageset.infomap[p]

    size_x, size_y = info.size

    img = pil_image.open(imageset.filename(p))

    # Determine the region of the original image we want to capture based on the bounding box.
    if info.bb is None:
        crop_margin = 0.0
        x0 = 0
        y0 = 0
        x1 = size_x
        y1 = size_y
    else:
        crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy
        x0, y0, x1, y1 = info.bb

    if info.rotate:
        x0, y0, x1, y1 = size_x - x1, size_y - y1, size_x - x0, size_y - y0
        img = img.rotate(180)

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

    # Transform image to black and white and comvert to numpy array
    img = img_to_array(img.convert('L'))

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


def getGlobals():
    return Globals()


def getTrainData(filename, test=None):
    tagged = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for row in reader:
            tagged[row[0]] = row[1]

    if test is not None:
        tagged = {k: tagged[k] for k in list(tagged)[:test]}
    return tagged


def getMappings(name):
    return deserialize(name, "mappings")


def getImageSet(name):
    return deserialize(name, "imageset")


def prepImageSet(name, datadir, images, useCache=True):
    imageset = ImageSet(datadir)

    for imagename in tqdm(images, desc="Image Info"):
        info = ImageInfo()
        img = pil_image.open(imageset.filename(imagename))

        info.size = img.size
        info.hash = phash(img)

        imageset.infomap[imagename] = info

    h2ps = {}
    for imagename, info in imageset.infomap.items():
        if info.hash not in h2ps:
            h2ps[info.hash] = []
        if imagename not in h2ps[info.hash]:
            h2ps[info.hash].append(imagename)

    # Two phash values are considered duplicate if, for all associated image pairs:
    # 1) They have the same mode and size;
    # 2) After normalizing the pixel to zero mean and variance 1.0, the mean square error does not exceed 0.1
    def match(h1, h2):
        for p1 in h2ps[h1]:
            for p2 in h2ps[h2]:
                i1 = pil_image.open(imageset.filename(p1))
                i2 = pil_image.open(imageset.filename(p2))
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

    # Find all distinct phash values
    hs = list(h2ps.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    for i, h1 in enumerate(tqdm(hs, desc="Hash Similarities")):
        for h2 in hs[:i]:
            if h1 - h2 <= 6 and match(h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2:
                    s1, s2 = s2, s1
                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for imagename, info in imageset.infomap.items():
        hash = str(info.hash)
        if hash in h2h:
            hash = h2h[hash]
        imageset.infomap[imagename] = hash

    serialize(imageset, name, "imageset")
    return imageset


def getTrainingHashes(w2hs):
    # Find the list of training images, keep only whales with at least two images.
    train = []  # A list of training image ids
    for hs in w2hs.values():
        if len(hs) > 1:
            train += hs
    return train


def prepMappings(name, imageset, tagged):
    mappings = Mappings()

    h2ps = {}
    for p, info in imageset.infomap.items():
        if info.hash not in h2ps:
            h2ps[info.hash] = []
        if p not in h2ps[info.hash]:
            h2ps[info.hash].append(p)

    # For each images id, select the prefered image
    def prefer(ps):
        if len(ps) == 1:
            return ps[0]
        best_p = ps[0]
        best_s = imageset.infomap[best_p].size
        for i in range(1, len(ps)):
            p = ps[i]
            s = imageset.infomap[p].size
            if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
                best_p = p
                best_s = s
        return best_p

    mappings.h2p = {}
    for h, ps in h2ps.items():
        mappings.h2p[h] = prefer(ps)

    h2ws = {}
    for p, w in tagged.items():
        h = imageset.infomap[p].hash
        if h not in h2ws:
            h2ws[h] = []
        if w not in h2ws[h]:
            h2ws[h].append(w)

    for h, ws in h2ws.items():
        if len(ws) > 1:
            h2ws[h] = sorted(ws)

    w2hs = {}
    for h, ws in h2ws.items():
        if len(ws) == 1:  # Use only unambiguous pictures
            w = ws[0]
            if w not in w2hs:
                w2hs[w] = []
            if h not in w2hs[w]:
                w2hs[w].append(h)

    for w, hs in w2hs.items():
        if len(hs) > 1:
            w2hs[w] = sorted(hs)

    mappings.h2ws = h2ws
    mappings.w2hs = w2hs

    serialize(mappings, name, "mappings")
    return mappings
