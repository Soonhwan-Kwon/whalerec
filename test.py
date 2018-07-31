import argparse

import utils


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action="store", type=int)  # Number of records to test with
#
# TODO: I think this will become the directory where the model and any supporting pickle files are stored.
#
parser.add_argument('-d', '--datadir', dest='datadir')
parser.add_argument('-D' '--images_dir', dest='imagedir')
parser.add_argument('-f', '--file', dest="file")
args = parser.parse_args()

config = utils.getConfig(args.datadir)

utils.debug_var("h2ps", config.h2ps)
utils.debug_var("h2p", config.h2p)
utils.debug_var("h2ws", config.h2ws)
utils.debug_var("w2hs", config.w2hs)
utils.debug_var("p2h", config.p2h)

# Find the list of training images, keep only whales with at least two images.
train = []  # A list of training image ids
for hs in config.w2hs.values():
    if len(hs) > 1:
        train += hs

utils.debug_var("train", train)


# known
utils.debug_var("known", list(config.h2ws.keys()))
utils.debug_var("known_sorted", sorted(list(config.h2ws.keys())))
