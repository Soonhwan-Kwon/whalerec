# Kaggle Playground

## Info

This is the winning entry by [Martin Piotte](https://www.kaggle.com/martinpiotte) to the [Humpback Whale Identification Challenge Kaggle Competition](https://www.kaggle.com/c/whale-categorization-playground/data). It has been reformmatted by [crowmagnumb](https://github.com/crowmagnumb)

Originally the train.csv file had a bunch of entries tagged as "new_whale". I think this was a mistake on Kaggle's part as Ted gave them a bunch of unidentified whales and they must have put some of them in the training set which makes no sense. Martin culled them out. I culled them from the csv file with ...

    cat trainORIG.csv | grep -v new_whale > train.csv

... so that we didn't have to deal with them in code because as I convert this we won't have that issue.

## Install

UPDATE: Don't need pandas anymore, so just install numpy. Of course installing pandas is still fine,
we just won't be using it.

    pip install pandas

which also installs numpy.

    pip install matplotlib
    pip install keras
    pip install tensorflow --upgrade
    pip install --user tqdm
    pip install --user imagehash
    pip install --user argparse

    #
    # There is a built-in slower alternative if this install causes issues.
    #
    pip install --user lap

## On AWS instance model-trainer

### Install NVidia drivers

Found this info [here](https://yangcha.github.io/CUDA90/)

```
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo apt-get update
sudo apt-get install cuda=9.0.176-1
sudo apt-get install libcudnn7-dev
sudo apt-get install libnccl-dev
```

Add to the end of your .bashrc file ...

```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Now reboot to get the load the NVidia drivers.

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

### Verify GPU Usage

    nvidia-smi

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

*   Isnt' the following line in id.py always going to create the same results? Cache?

    fknown = model.branch.predict_generator(FeatureGen(config, utils.hashes2images(known)), max_queue_size=20, workers=10, verbose=0)
