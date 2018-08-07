import sys
import glob

import argparse

import utils
import modelUtils

new_whale = 'new_whale'


def perform_id(h2ws, score, threshold, data):
    """
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    # TODO: Check if this needs to be sorted. Saves time?
    known = sorted(list(h2ws.keys()))

    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    for i, p in enumerate(tqdm(data)):
        t = []
        s = set()
        a = score[i, :]
        for j in list(reversed(np.argsort(a))):
            h = known[j]
            if a[j] < threshold and new_whale not in s:
                pos[len(t)] += 1
                s.add(new_whale)
                t.append(new_whale)
                if len(t) == 5:
                    break

            for w in h2ws[h]:
                assert w != new_whale
                if w not in s:
                    if a[j] > 1.0:
                        vtop += 1
                    elif a[j] >= threshold:
                        vhigh += 1
                    s.add(w)
                    t.append(w)
                    if len(t) == 5:
                        break
            if len(t) == 5:
                break
        if new_whale not in s:
            pos[5] += 1
        assert len(t) == 5 and len(s) == 5

        print(p + ',' + ' '.join(t[:5]) + '\n')
    # return vtop, vhigh, pos


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action="store", type=int)  # Number of records to test with
parser.add_argument('-n', '--name', dest='name')
parser.add_argument('-D' '--images_dir', dest='imagedir')
parser.add_argument('-f', '--file', dest="file")
args = parser.parse_args()

imageset = utils.getImageSet(args.name)
mappings = utils.getMappings(args.name)

model = modelUtils.get_standard()

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

submitImageset = utils.getImageSet(args.imagedir, submit, False)

if model is None:
    print("Model does not exist! Exiting!")
    sys.exit()

# TODO: Save fknown in model directory as pickle so that we only have to run this once.
# Again, do the keys have to be sorted here? Saves time? If we cache it I guess that doesn't matter
trainedData = utils.hashes2images(mappings.h2p, sorted(list(mappings.h2ws.keys())))
fknown = model.branch.predict_generator(FeatureGen(imageset, trainedData), max_queue_size=20, workers=10, verbose=0)

fsubmit = model.branch.predict_generator(FeatureGen(submitImageset, submit), max_queue_size=20, workers=10, verbose=0)
score = model.head.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
score = modelUtils.score_reshape(score, fknown, fsubmit)

perform_id(mappings.h2ws, score, 0.99, submit)
