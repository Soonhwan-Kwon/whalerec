import sys
import json
import argparse

import modelUtils

parser = argparse.ArgumentParser()
parser.add_argument("--serialize", action="store_true")
parser.add_argument('-s', '--stage', action="store", type=int)  # Number of steps to read the model at
parser.add_argument('-m', '--min_matches', default=0, action="store", type=int)  # Number of steps to read the model at
parser.add_argument('-r', '--refset', dest='refset')
parser.add_argument('-D' '--imgdir', dest='imgdir')
parser.add_argument('-o', '--output')
parser.add_argument('--threshold', defaul=0.99, type=float)
args = parser.parse_args()

model, mappings, fknown = modelUtils.get_refset_info(args.refset, args.stage)

modelUtils.perform_id(model, mappings, fknown, args.refset, args.imgdir, args.serialize, args.threshold, args.min_matches)

json_data = json.dumps(results)
if args.output is None:
    print(json_data)
else:
    with open(args.output, 'w') as output:
        output.write(json_data)
