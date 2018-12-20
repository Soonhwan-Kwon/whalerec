# Kaggle Playground

## Info

This is the winning entry by [Martin Piotte](https://www.kaggle.com/martinpiotte) to the [Humpback Whale Identification Challenge Kaggle Competition](https://www.kaggle.com/c/whale-categorization-playground/data). It has been reformatted by [crowmagnumb](https://github.com/crowmagnumb)

Originally the train.csv file had a bunch of entries tagged as "new_whale". I think this was a mistake on Kaggle's part as Ted gave them a bunch of unidentified whales and they must have put some of them in the training set which makes no sense. Martin culled them out. I culled them from the csv file with ...

    cat trainORIG.csv | grep -v new_whale > train.csv

... so that we didn't have to deal with them in code because as I convert this we won't have that issue.

### Install tensorflow and other python packages

    # pipenv install pandas
    pipenv install numpy

    pipenv install tqdm
    pipenv install imagehash
    pipenv install argparse

    pipenv install tensorflow-gpu
    pipenv install keras
    pipenv install keras_tqdm

### Run

    python id.py -d ~/kaggle

... or this if you want to ignore warnings ...

    python -W ignore id.py -d ~/kaggle

## TODO

Build tagged dict automatically in production code using existing directory structure. Actually can just skip that step and go straight to getting w2hs or h2ws or whatever it is we need.

    set1/whale1/file1
    set1/whale1/file2
    set1/whale1/fileN
    set1/whale2/file1
    ...
    setN/whaleN/fileN

Filenames will be path starting setN/... and not just the name of the image_file.

Can create separate models with each set or one big model for all the sets.

Isn't the following line in id.py always going to create the same results? Cache?
At the very least we should cache the entire imageset and mappings objects. And we don't/won't need to load all the tagged dictionary.

    fknown = model.branch.predict_generator(FeatureGen(imageset, utils.hashes2images(mappings.h2p, known))), max_queue_size=20, workers=10, verbose=0)
