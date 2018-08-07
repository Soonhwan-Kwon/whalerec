import argparse

import utils
import modelUtils


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name')
parser.add_argument('-t', '--test', action='store_true')
parser.add_argument('-d', '--datadir', dest='datadir')
parser.add_argument('-f' '--file', dest="csvfile")
args = parser.parse_args()

setname = args.name

imageset = utils.getImageSet(setname)
mappings = utils.getMappings(setname)

if imageset is None or mappings is None:
    tagged = utils.getTrainData(args.csvfile)
    if imageset is None:
        imageset = utils.prepImageSet(setname, args.datadir, list(tagged.keys()))
    if mappings is None:
        mappings = utils.prepMappings(setname, imageset, tagged)

modelUtils.make_standard(setname, imageset, mappings, args.test)
