import argparse

import modelUtils

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--stage', action="store", type=int)  # Number of steps to read the model at
parser.add_argument('-r', '--refset', required=True)
args = parser.parse_args()

fknown = modelUtils.make_fknown(args.refset, args.stage)
modelUtils.serialize_fknown(args.refset, fknown, args.stage)
