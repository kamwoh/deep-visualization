import os

import cv2
import numpy as np

import utils


def main():
    from vgg16 import model
    from imagenet_classes import class_names

    display_weight = False

    dirname = os.path.dirname(os.path.abspath(__file__))
    layer_idx = 18

    img_disp = cv2.imread('{}/images/car.jpg'.format(dirname))
    img_disp = cv2.resize(img_disp, (224, 224))
    img = img_disp[np.newaxis, :, :, :]
    img = img.astype(np.float32)
    img = img - np.array([103.939, 116.779, 123.68])  # bgr

    for i, layer in enumerate(model.layers):
        print i, layer

    out = utils.get_layers(img, model, layer_idx)
    out = np.transpose(out, (3, 1, 2, 0))
    out = utils.normalize_image(out)
    disp = utils.combine_and_fit(out, factor=5, is_conv=True)

    if display_weight:
        weight = model.get_weights()[
            (layer_idx - 1) * 2]  # minus 1 because first layer in model is Input, multiply 2 because to skip bias
        weight = utils.normalize_weights(weight, 'conv')
        weight = np.transpose(weight, (3, 0, 1, 2))
        weight_disp = utils.combine_and_fit(weight, factor=10)
        cv2.imshow('weight_disp', weight_disp)

    cv2.imshow('input', img_disp)
    cv2.imshow('disp', disp)

    prob = model.predict(img)[0]
    preds = (np.argsort(prob)[::-1])[:5]

    for p in preds:
        print class_names[p], prob[p]

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
