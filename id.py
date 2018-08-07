import sys
import glob

import argparse

import globals
import utils
import modelUtils

from modelUtils import FeatureGen, ScoreGen

new_whale = 'new_whale'


def perform_id(h2ws, score, threshold, images):
    """
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    # TODO: Check if this needs to be sorted. Saves time?
    known = sorted(list(h2ws.keys()))

    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    for ii, img in enumerate(tqdm(images)):
        whalelist = []
        whaleset = set()
        scores = score[ii, :]
        for jj in list(reversed(np.argsort(scores))):
            hash = known[jj]
            if scores[jj] < threshold and new_whale not in whaleset:
                pos[len(whalelist)] += 1
                whaleset.add(new_whale)
                whalelist.append(new_whale)
                if len(whalelist) == 5:
                    break

            for whale in h2ws[hash]:
                assert whale != new_whale
                if whale not in whaleset:
                    if scores[jj] > 1.0:
                        vtop += 1
                    elif scores[jj] >= threshold:
                        vhigh += 1
                    whaleset.add(whale)
                    whalelist.append(whale)
                    if len(whalelist) == 5:
                        break
            if len(whalelist) == 5:
                break
        if new_whale not in whaleset:
            pos[5] += 1
        assert len(whalelist) == 5 and len(whaleset) == 5

        print(img + ',' + ' '.join(whalelist[:5]) + '\n')
    # return vtop, vhigh, pos


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action="store_true")
parser.add_argument('-s', '--stage', action="store", type=int)  # Number of steps to read the model at
parser.add_argument('-n', '--name', dest='name')
parser.add_argument('-D' '--images_dir', dest='imagedir')
parser.add_argument('-f', '--file', dest="file")
args = parser.parse_args()

setname = args.name

model = modelUtils.get_standard(setname, args.stage)

if model is None:
    print("Model does not exist! Exiting!")
    sys.exit()

# filename = datadir + "/sample_submission.csv"
# submit = []
# with open(filename, newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader, None)  # skip the headers
#     for row in reader:
#         submit.append(row[0])
if args.file:
    submit = [args.file]
else:
    submit = []
    submit = glob.glob(args.imagedir + "/*", recursive=True)

#
# If we are testing then we may have wanted to save the prep work so that we can
# repeat this iding very quickly. So we first check to see if we have already created
# the imageset pickle
#
if args.test:
    submitImageset = utils.deserialize(args.imagedir, globals.IMAGESET)
    if submitImageset is None:
        submitImageset = utils.prepImageSet(args.imagedir, submit)
        utils.serialize(args.imagedir, globals.IMAGESET)
else:
    submitImageset = utils.prepImageSet(args.imagedir, submit)

mappings = utils.getMappings(setname)

# Save fknown in model directory as pickle so that we only have to run this once.
# Again, do the keys have to be sorted here? Saves time? If we cache it I guess that doesn't matter
# Now run prep_id.py first on the trained model before running any id requests.
# UPDATE: Getting None for deseriazilation. Switching back.

# fknown = modelUtils.deserialize_fknown(setname, args.stage)
fknown = modelUtils.make_fknown2(setname, model, mappings)

fsubmit = model.branch.predict_generator(FeatureGen(submitImageset, submit), max_queue_size=20, workers=10, verbose=0)
score = model.head.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
score = modelUtils.score_reshape(score, fknown, fsubmit)


perform_id(mappings.h2ws, score, 0.99, submit)
