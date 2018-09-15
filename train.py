import argparse
import csv
import os

import globals
import utils
import modelUtils


parser = argparse.ArgumentParser()
parser.add_argument("--imgdir", required=True)
parser.add_argument('-r', '--refset', required=True)

parser.add_argument("--csvfile", default=None)

parser.add_argument('--ingest-type', dest='ingest_type', default='second_dash',
                    choices=['named_folders', 'second_dash', 'allied'])

parser.add_argument('-t', '--test', action='store_true')
args = parser.parse_args()

refset = args.refset
imageset = utils.getImageSet(refset)
mappings = utils.getMappings(refset)

if imageset is None or mappings is None:
    namedfiles = []
    if args.csvfile is None:
        if args.ingest_type == 'named_folders':
            def get_name_from_folder(filepath):
                return os.path.basename(os.path.dirname(filepath))
            extract_name = get_name_from_folder
        elif args.ingest_type == 'second_dash':
            def second_dash(filepath):
                bits = os.path.basename(filepath).split("-")
                return bits[0] + "-" + bits[1]
            extract_name = second_dash
        elif args.ingest_type == 'allied':
            def allied(filepath):
                return os.path.basename(filepath).split("sn")[0]
            extract_name = allied

        IMG_EXTS = ['*.[jJ][pP][gG]', '*.[jJ][pP][eE][gG]', '*.[pP][nN][gG]']
        for ext in IMG_EXTS:
            for image in [y for x in os.walk(os.path.realpath(args.imgdir)) for y in glob.iglob(os.path.join(x[0], ext))]:
                namedfiles.append((image[len(args.imgdir) + 1:], extract_name(image)))
    else:
        with open(args.csvfile, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip the headers
            for row in reader:
                namedfiles.append((row[0], row[1]))

    if imageset is None:
        imageset = utils.prepImageSet(args.imgdir, [namedfile[0] for namedfile in namedfiles])
        utils.serialize_set(refset, imageset, globals.IMAGESET)

    if mappings is None:
        mappings = utils.prepMappings(imageset, namedfiles)
        utils.serialize_set(refset, mappings, globals.MAPPINGS)


modelUtils.make_standard(refset, imageset, mappings, args.test)
