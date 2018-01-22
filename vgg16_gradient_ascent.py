import os

import cv2
import tensorflow as tf
from keras.applications import VGG16

import utils


def main():
    g = tf.Graph()
    sess = tf.Session(graph=g)
    with g.as_default():
        with sess.as_default():
            model = VGG16()

    dirname = os.path.dirname(os.path.abspath(__file__))
    layer_idx = 17

    for i, layer in enumerate(model.layers):
        print(i, layer.output)

    img = utils.generate_random_image(model.layers[layer_idx].output.get_shape().as_list()[3],
                                      [224, 224, 3])

    for out in utils.deepdream(img, model, layer_idx, g=g, sess=sess):
        out = utils.visstd(out, per_image=True)
        out = utils.combine_and_fit(out, is_deconv=True, disp_w=1000)
        out = utils.to_255(out)

        cv2.imshow('deepdream', out)
        cv2.waitKey(1) & 0xFF

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
