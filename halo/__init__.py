from halo.iou import compute_iou
import halo.config as cfg
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

for dirPath in (cfg.POSITIVE_PATH, cfg.NEGATIVE_PATH):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

imagePaths = list(paths.list_images(cfg.ORIG_IMAGES))

totalPositives = 0
totalNegatives = 0

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}...".format(i + 1, len(imagePaths)))

    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]

    annotPath = os.path.sep.join([cfg.ORIG_ANNOTS, "{}.xml".format(filename)])
    contents = open(annotPath).read()

    soup = BeautifulSoup(contents, "html.parser")
    gtBoxes = []

    w = int(soup.find("width").string)
    h = int(soup.find("height").string)

    for o in soup.find_all("object"):
        label = o.find("name").string
        xMin = int(o.find("xmin").string)
        yMin = int(o.find("ymin").string)
        xMax = int(o.find("xmax").string)
        yMax = int(o.find("ymax").string)

        xMin = max(0, xMin)
        yMin = max(0, yMin)
        xMax = min(w, xMax)
        yMax = min(h, yMax)

        gtBoxes.append((xMin, yMin, xMax, yMax))

    image = cv2.imread(imagePath)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposedRects = []

    for (x, y, w, h) in rects:
        proposedRects.append((x, y, x + w, y + h))

    positiveROIs = 0
    negativeROIs = 0

    for proposedRect in proposedRects[:cfg.MAX_PROPOSALS]:
        (propStartX, propStartY, propEndX, propEndY) = proposedRect

        for gtBox in gtBoxes:
            iou = compute_iou(gtBox, proposedRect)
            (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

            roi = None
            outputPath = None

            if iou > 0.7 and positiveROIs <= cfg.MAX_POSITIVE:
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalPositives)
                outputPath = os.path.sep.join([cfg.POSITVE_PATH, filename])

                positiveROIs += 1
                totalPositives += 1

            fullOverlap = propStartX >= gtStartX
            fullOverlap = fullOverlap and propStartY >= gtStartY
            fullOverlap = fullOverlap and propEndX <= gtEndX
            fullOverlap = fullOverlap and propEndY <= gtEndY

            if not fullOverlap and iou < 0.05 and negativeROIs <= cfg.MAX_NEGATIVE:
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalNegatives)
                outputPath = os.path.sep.join([cfg.NEGATIVE_PATH, filename])

                negativeROIs += 1
                totalNegatives += 1

            if roi is not None and outputPath is not None:
                roi = cv2.resize(roi, cfg.INPUT_DIMS,
                                 interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)
