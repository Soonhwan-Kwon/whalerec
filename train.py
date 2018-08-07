import argparse

import utils
import modelUtils


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name')
parser.add_argument('-t', '--test', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
parser.add_argument('-f' '--file', dest="csvfile")
args = parser.parse_args()

imageset = utils.getImageSet(args.name)
mappings = utils.getMappings(args.name)

if imageset is None or mappings is None:
    tagged = utils.getTrainData(args.csvfile)
    if imageset is None:
        imageset = utils.prepImageSet(args.name, args.datadir, list(tagged.keys()))
    if mappings is None:
        mappings = utils.prepMappings(args.name, imageset, tagged)

modelUtils.make_standard(imageset, mappings, args.test)
