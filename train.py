import argparse

import utils
import modelUtils


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
args = parser.parse_args()

globals = utils.getGlobals()
tagged = utils.getTrainData(args.datadir)
config = utils.getConfig(args.datadir, list(tagged.keys()))
mappings = utils.getMappings(config, tagged)

modelUtils.make_standard(globals, config, mappings, args.test)
