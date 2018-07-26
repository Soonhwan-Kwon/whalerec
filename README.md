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

## RUN

    python -W ignore id.py -d ~/tmp/kaggle --test

## TODO

*   Instead of using the bounding box (p2bb), use instead the entire size of the image. p2size?
    OK, done, BUT why are the resized images cropped badly? The tips are cut off.

-   Check into github!
