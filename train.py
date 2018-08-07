import argparse

import utils
import modelUtils


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
args = parser.parse_args()

globals = utils.getGlobals()
tagged = utils.getTrainData(args.datadir)
imageset = utils.getImageSet(args.datadir, list(tagged.keys()))
mappings = utils.getMappings(imageset, tagged)

modelUtils.make_standard(globals, imageset, mappings, args.test)
