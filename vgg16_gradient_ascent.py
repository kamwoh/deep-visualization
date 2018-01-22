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

    n = 4
    channel = 0
    layer_idx = 17

    for i, layer in enumerate(model.layers):
        print(i, layer.output)

    img = utils.generate_random_image(n,
                                      [224, 224, 3])

    for out in utils.deepdream(img, model, layer_idx, channel, iterations=100, g=g, sess=sess):
        out_mean = utils.visstd(out, per_image=True)
        out_mean = utils.combine_and_fit(out_mean, is_deconv=True, disp_w=1000)
        out_mean = utils.to_255(out_mean)

        out_minmax = utils.normalize(out, per_image=True)
        out_minmax = utils.combine_and_fit(out_minmax, is_deconv=True, disp_w=1000)
        out_minmax = utils.to_255(out_minmax)

        cv2.imshow('deepdream1', out_mean)
        cv2.imshow('deepdream2', out_minmax)
        cv2.waitKey(1) & 0xFF

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
