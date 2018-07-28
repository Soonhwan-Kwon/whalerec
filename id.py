import sys
import argparse

import utils
import model


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
args = parser.parse_args()


config = utils.getConfig(args.datadir, args.test)
data = utils.getData(config)

#
# THIS CAN TOTALLY BE MOVED UP THE LINE BEFORE TO JUST BEFORE THE MAKE STANDARD
#
data.model, data.branch_model, data.head_model = model.build(config.img_shape, 64e-5, 0)
# head_model.summary()
# branch_model.summary()

model.make_standard(config, data)
