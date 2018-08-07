import argparse

import modelUtils

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--stage', action="store", type=int)  # Number of steps to read the model at
parser.add_argument('-n', '--name', dest='name')
args = parser.parse_args()

setname = args.name

fknown = modelUtils.make_fknown(setname, args.stage)
modelUtils.serialize_fknown(setname, fknown, args.stage)
