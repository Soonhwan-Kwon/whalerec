import sys
import glob

import argparse

import utils
import modelUtils

new_whale = 'new_whale'


def perform_id(h2ws, threshold, data):
    """
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
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
#
# TODO: I think this will become the directory where the model and any supporting pickle files are stored.
#
parser.add_argument('-d', '--datadir', dest='datadir')
parser.add_argument('-D' '--images_dir', dest='imagedir')
parser.add_argument('-f', '--file', dest="file")
args = parser.parse_args()

globals = utils.getGlobals()
config = utils.getConfig(args.datadir, args.test)

known = sorted(list(config.h2ws.keys()))

model = modelUtils.get_standard(globals)

if args.file:
    submit = [args.file]
else:
    submit = []
    submit = glob.glob(args.imagedir + "/*", recursive=True)

print(known, submit)

if model is None:
    print("Model does not exist! Exiting!")
    sys.exit()

# Evaluate the model.
fknown = model.branch.predict_generator(FeatureGen(globals, config, utils.hashes2images(known)), max_queue_size=20, workers=10, verbose=0)
fsubmit = model.branch.predict_generator(FeatureGen(globals, config, submit), max_queue_size=20, workers=10, verbose=0)
score = model.head.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
score = modelUtils.score_reshape(score, fknown, fsubmit)

# Generate the subsmission file.
perform_id(h2ws, 0.99, submit)
