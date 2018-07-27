import random

import utils

from keras.utils import Sequence
import numpy as np
from keras import backend as K

# First try to use lapjv Linear Assignment Problem solver as it is much faster.
# At the time I am writing this, kaggle kernel with custom package fail to commit.
# scipy can be used as a fallback, but it is too slow to run this kernel under the time limit
# As a workaround, use scipy with data partitioning.
# Because algorithm is O(n^3), small partitions are much faster, but not what produced the submitted solution
try:
    from lap import lapjv
    segment = False
except ImportError:
    print('Module lap not found, emulating with much slower scipy.optimize.linear_sum_assignment')
    segment = True
    from scipy.optimize import linear_sum_assignment


class TrainingData(Sequence):
    def __init__(self, config, train, score, steps=1000, batch_size=32):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()

        self.config = config
        self.train = train
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        for ts in config.w2ts.values():
            idxs = [config.t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + self.config.img_shape, dtype=K.floatx())
        b = np.zeros((size,) + self.config.img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = utils.read_cropped_image(self.config, self.match[j][0], True)
            b[i, :, :, :] = utils.read_cropped_image(self.config, self.match[j][1], True)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = utils.read_cropped_image(self.config, self.unmatch[j][0], True)
            b[i + 1, :, :, :] = utils.read_cropped_image(self.config, self.unmatch[j][1], True)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        return [a, b], c

    def on_epoch_end(self):
        if self.steps <= 0:
            return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        if segment:
            # Using slow scipy. Make small batches.
            # Because algorithm is O(n^3), small batches are much faster.
            # However, this does not find the real optimum, just an approximation.
            tmp = []
            batch = 512
            # EDITED: Added self in front of score for this and the next line. The others were already self? Correct?
            # Could go back to using the global one which I guess it was using? Should be the same because the shape
            # should be the same BUT I don't think the global is defined when it first hits here. At least in the debug
            # version.
            for start in range(0, self.score.shape[0], batch):
                end = min(self.score.shape[0], start + batch)
                _, x = linear_sum_assignment(self.score[start:end, start:end])
                tmp.append(x + start)
            x = np.concatenate(tmp)
        else:
            _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in self.config.w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d):
                    break
            for ab in zip(ts, d):
                self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((self.train[i], self.train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(self.train) and len(self.unmatch) == len(self.train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size
