import os

ORIG_PATH = os.path.sep.join(["datasets", "original"])
ORIG_IMAGES = os.path.sep.join([ORIG_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_PATH, "annotations"])

GEN_PATH = os.path.sep.join(["datasets", "generated"])
POSITIVE_PATH = os.path.sep.join([GEN_PATH, "raccoon"])
NEGATIVE_PATH = os.path.sep.join([GEN_PATH, "no_raccoon"])

MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

MAX_POSITIVE = 30
MAX_NEGATIVE = 10

INPUT_DIMS = (224, 224)

MODEL_PATH = "raccoon_detector.h5"
ENCODER_PATH = "raccoon_encoder.pickle"

MIN_PROBA = 0.99
