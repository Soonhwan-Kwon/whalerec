import argparse

import utils


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name')
args = parser.parse_args()

imageset = utils.getImageSet(args.name)
mappings = utils.getMappings(args.name)

utils.debug_var("h2ps", mappings.h2ps)
utils.debug_var("h2p", mappings.h2p)
utils.debug_var("h2ws", mappings.h2ws)
utils.debug_var("w2hs", mappings.w2hs)

# Find the list of training images, keep only whales with at least two images.
train = []  # A list of training image ids
for hs in mappings.w2hs.values():
    if len(hs) > 1:
        train += hs

utils.debug_var("train", train)


# known
utils.debug_var("known", list(mappings.h2ws.keys()))
utils.debug_var("known_sorted", sorted(list(mappings.h2ws.keys())))
