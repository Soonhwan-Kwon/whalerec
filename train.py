import argparse

import utils
import modelUtils


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
args = parser.parse_args()


config = utils.getConfig(args.datadir, args.test)

modelUtils.make_standard(config, args.test)
