# Kaggle Playground

## Info

This is the winning entry by [Martin Piotte](https://www.kaggle.com/martinpiotte) to the [Humpback Whale Identification Challenge Kaggle Competition](https://www.kaggle.com/c/whale-categorization-playground/data). It has been reformmatted by [crowmagnumb](https://github.com/crowmagnumb)

## Install

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

    pipenv install pandas
    pipenv install matplotlib
    pipenv install keras
    pipenv install tensorflow
    pipenv install tqdm
    pipenv install imagehash
    pipenv install argparse

## RUN

    python -W ignore id.py -d ~/tmp/kaggle --test

## TODO

*   Check with Martin about the global score thing - I think its probably fine.
