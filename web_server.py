import tornado.ioloop
import tornado.web
import json
import argparse

import modelUtils

modelInfos = {}


model, mappings, fknown = modelUtils.get_refset_info(args.refset, args.stage)

modelUtils.perform_id(model, mappings, fknown, args.imgdir, args.serialize, args.threshold, args.min_matches)


def get_model_info(refset):
    if refset in modelInfos:
        return modelInfos[refset]
    return set_model_info(refset)


def set_model_info(refset):
    modelInfo = modelUtils.get_refset_info(refset)
    modelInfos[refset] = modelInfo
    return modelInfo


class IdentifyHandler(tornado.web.RequestHandler):
    def post(self):
        refset = self.get_argument("refset")
        imgdir = self.get_argument("imgdir")
        imageset = self.get_argument("imageset")

        model, mappings, fknown = get_model_info(refset)
        results = modelUtils.perform_id(model, mappings, fknown, imgdir)
        self.write(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    set_model_info("humpbacks-test")

    # can also use autoreload=True for auto-checking for code changes if you
    # don't want other debug features
    app = tornado.web.Application([
        (r"/identify", IdentifyHandler),
    ], debug=args.debug)

    app.listen(8889)
    tornado.ioloop.IOLoop.current().start()
