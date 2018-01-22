import os

import cv2
import numpy as np
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

    img_disp = cv2.imread('{}/images/woh.png'.format(dirname))
    img_disp = cv2.resize(img_disp, (224, 224))
    img = img_disp[np.newaxis, :, :, :]
    img = img.astype(np.float32)
    img = img - np.array([103.939, 116.779, 123.68])  # bgr

    # print(deepdream_visualization(None, {model.layers[0].input: img}, 'block1_conv1/Relu', [0,1]))

    for out in utils.deepdream(img, model, layer_idx, g=g, sess=sess):
        out = utils.normalize_image(out, per_image=True)
        out = utils.combine_and_fit(out, is_deconv=True, disp_w=4000)
        out = utils.to_255(out)

        cv2.imshow('deepdream', out)
        cv2.waitKey(1) & 0xFF
        # cv2.imwrite('{}_deconv_out.png'.format(model.layers[layer_idx].name), out)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
