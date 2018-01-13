import os

import cv2
import numpy as np

import utils


def main():
    """

    :return:
    """
    from vgg16 import model

    dirname = os.path.dirname(os.path.abspath(__file__))

    img = cv2.imread('{}/images/laska.png'.format(dirname))
    img = cv2.resize(img, (224, 224))
    img = img[np.newaxis, :, :, :]
    img = img.astype(np.float32)
    img = img - np.array([103.939, 116.779, 123.68])  # bgr

    for i, layer in enumerate(model.layers):
        print i, layer

    out = utils.get_layers(img, model, 1)
    out = np.transpose(out, (3, 1, 2, 0))

    out = utils.normalize_image(out)
    disp = utils.combine_and_fit(out, factor=0.5, is_layer=True)

    # out = np.square(out, 0)

    cv2.imshow('test', disp)
    cv2.waitKey(0)

    #
    # for p in preds:
    #     print class_names[p], prob[p]


if __name__ == '__main__':
    main()
