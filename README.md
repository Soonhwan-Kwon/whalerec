# Kaggle Playground

## Info

This is the winning entry by [Martin Piotte](https://www.kaggle.com/martinpiotte) to the [Humpback Whale Identification Challenge Kaggle Competition](https://www.kaggle.com/c/whale-categorization-playground/data). It has been reformatted by [crowmagnumb](https://github.com/crowmagnumb)

Originally the train.csv file had a bunch of entries tagged as "new_whale". I think this was a mistake. We gave them a bunch of unidentified whales and some of them must have gotten into the training set. Martin culled them out. I culled them from the csv file with ...

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

### Train

For kaggle competition ...

    pipenv run python train.py -r <refset> --csvfile <csvfile>

<refset> is just the name you want to identify it with. e.g. "humpbacks" and <csvfile> is the file containing the training data mapping of whale to image file.

For referencesets going forward ...

    pipenv run python train.py -r <refset> --imgdir <imgdir> --ingest-type <named_folders | second_dash | sn>

... where ingest-type is one of three types of which named_folders is our main one. In this type images are grouped into sub-directories of <imgdir> whose name is the name of the whale. In others the name of the whale is embedded in the image name and the images are in <imgdir> or any sub-dir of <imgdir>.

### Run Identification

    pipenv run python id.py -r <refset> -D <img_dir>

... where <img_dir> is the directory containing the images that you want to try and identify. In production mode images to be ID'ed are uploaded into a temp directory and then id'ed by the above command.

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
