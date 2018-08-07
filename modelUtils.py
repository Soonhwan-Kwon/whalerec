import os
import random

import utils

from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import Sequence
from keras_tqdm import TQDMCallback

from tqdm import tqdm
import numpy as np

from trainUtils import TrainingData
from globals import IMG_SHAPE


class Execution(object):
    def __init__(self):
        self.steps = 0
        self.histories = []
        self.score = None


class WhaleRecModel(object):
    def __init__(self, siamese, branch, head):
        self.siamese = siamese
        self.branch = branch
        self.head = head


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, imageset, images, batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()

        self.imageset = imageset
        self.images = images
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.images) - start, self.batch_size)
        a = np.zeros((size,) + IMG_SHAPE, dtype=K.floatx())

        for i in range(size):
            a[i, :, :, :] = utils.read_cropped_image(self.imageset, self.images[start + i], False)
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self):
                self.progress.close()
        return a

    def __len__(self):
        return (len(self.images) + self.batch_size - 1) // self.batch_size


# A Keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only the upper triangular matrix of the cost matrix if y is None.
class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size,))
            self.iy = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self):
                self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size


def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build(lr, l2, activation='sigmoid'):

    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=IMG_SHAPE)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=IMG_SHAPE)
    img_b = Input(shape=IMG_SHAPE)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    siamese = Model([img_a, img_b], x)
    siamese.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

    return WhaleRecModel(siamese, branch_model, head_model)


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m


def make_fknown(setname, steps=None):
    mappings = utils.getMappings(setname)
    model = get_standard(setname, steps)
    return make_known2(setname, model, mappings)


def make_fknown2(setname, model, mappings):
    imageset = utils.getImageSet(setname)
    trainedData = utils.hashes2images(mappings.h2p, sorted(list(mappings.h2ws.keys())))
    # utils.debug_var("trainedData", trainedData)
    return model.branch.predict_generator(FeatureGen(imageset, trainedData), max_queue_size=20, workers=10, verbose=0)


def name_fknown(steps=None):
    if steps is None:
        return "fknown"
    return "fknown-%s" % str(steps)


def serialize_fknown(setname, fknown, steps=None):
    utils.serialize_set(setname, fknown, name_fknown(steps))


def deserialize_fknown(setname, steps=None):
    utils.deserialize_set(setname, name_fknown(steps))


def make_steps(imageset, mappings, model, execution, train, steps, ampl):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    # shuffle the training pictures
    random.shuffle(train)

    # Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    trainImages = utils.hashes2images(mappings.h2p, train)

    features = model.branch.predict_generator(FeatureGen(imageset, trainImages, verbose=1),
                                              max_queue_size=12,
                                              workers=6,
                                              verbose=0)
    score = model.head.predict_generator(ScoreGen(features, verbose=1), max_queue_size=12, workers=6, verbose=0)
    execution.score = score_reshape(score, features)

    # Train the model for 'steps' epochs
    history = model.siamese.fit_generator(
        TrainingData(imageset,
                     mappings,
                     train,
                     execution.score + ampl * np.random.random_sample(size=execution.score.shape),
                     steps=steps,
                     batch_size=32),
        initial_epoch=execution.steps,
        epochs=execution.steps + steps,
        max_queue_size=12,
        workers=6,
        verbose=0,
        callbacks=[TQDMCallback(leave_inner=True, metric_format='{value:0.3f}')]
    ).history

    execution.steps += steps
    print("STEPS: ", execution.steps)

    # Collect history data
    history['epochs'] = execution.steps
    history['ms'] = np.mean(execution.score)
    history['lr'] = get_lr(model.siamese)
    print(history['epochs'], history['lr'], history['ms'])
    execution.histories.append(history)


def get_model_file(setname, type, steps=None):
    filename = type + ".model"
    if steps is None:
        filename = type
    else:
        filename = "%s-%s" % (type, str(steps))
    return os.path.join(utils.set_directory(setname), filename + ".model")


def save_standard(setname, model, steps=None):
    filename = get_model_file(setname, "standard", steps)
    print("Saving model [" + filename + "]")
    model.siamese.save(filename)


def get_standard(setname, steps=None):
    filename = get_model_file(setname, "standard", steps)
    if os.path.isfile(filename):
        print("Loading model [" + filename + "]")
        model = build(64e-5, 0)
        tmp = load_model(filename)
        model.siamese.set_weights(tmp.get_weights())
        return model
    else:
        raise ValueError("Model file [" + filename + "] not found.")


def make_standard(setname, imageset, mappings, test=False):
    execution = Execution()

    train = utils.getTrainingHashes(mappings.w2hs)

    print("Training Images: ", len(train))
    if len(train) == 0:
        print("No data to train on! Exiting!")
        return

    random.shuffle(train)

    model = build(64e-5, 0)
    # model.head.summary()
    # model.branch.summary()

    # epoch -> 10
    if test:
        make_steps(imageset, mappings, model, execution, train, 1, 1000)
        save_standard(setname, model, execution.steps)

        make_steps(imageset, mappings, model, execution, train, 1, 100.0)
    else:
        make_steps(imageset, mappings, model, execution, train, 10, 1000)
        save_standard(setname, model, execution.steps)

        ampl = 100.0
        for _ in range(10):
            make_steps(imageset, mappings, model, execution, train, 5, ampl)
            ampl = max(1.0, 100**-0.1 * ampl)
        save_standard(setname, model, execution.steps)

        # epoch -> 150
        for _ in range(18):
            make_steps(imageset, mappings, model, execution, train, 5, 1.0)
        save_standard(setname, model, execution.steps)

        # epoch -> 200
        set_lr(model.siamese, 16e-5)
        for _ in range(10):
            make_steps(imageset, mappings, model, execution, train, 5, 0.5)
        save_standard(setname, model, execution.steps)

        # epoch -> 240
        set_lr(model.siamese, 4e-5)
        for _ in range(8):
            make_steps(imageset, mappings, model, execution, train, 5, 0.25)
        save_standard(setname, model, execution.steps)

        # epoch -> 250
        set_lr(model.siamese, 1e-5)
        for _ in range(2):
            make_steps(imageset, mappings, model, execution, train, 5, 0.25)
        save_standard(setname, model, execution.steps)

        # epoch -> 300
        weights = model.siamese.get_weights()

        model = build(64e-5, 0.0002)
        model.siamese.set_weights(weights)

        for _ in range(10):
            make_steps(imageset, mappings, model, execution, train, 5, 1.0)
        save_standard(setname, model, execution.steps)

        # epoch -> 350
        set_lr(model.siamese, 16e-5)
        for _ in range(10):
            make_steps(imageset, mappings, model, execution, train, 5, 0.5)
        save_standard(setname, model, execution.steps)

        # epoch -> 390
        set_lr(model.siamese, 4e-5)
        for _ in range(8):
            make_steps(imageset, mappings, model, execution, train, 5, 0.25)
        save_standard(setname, model, execution.steps)

        # epoch -> 400
        set_lr(model.siamese, 1e-5)
        for _ in range(2):
            make_steps(imageset, mappings, model, execution, train, 5, 0.25)

    save_standard(setname, model)
