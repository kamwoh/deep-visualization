import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # not enough memory, need to turn off gpu

import numpy as np
import cv2

import utils
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from keras.applications import VGG16


def main():
    @tf.RegisterGradient("GuidedRelu")
    def _GuidedReluGrad(op, grad):
        return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

    g = tf.Graph()
    sess = tf.Session(graph=g)
    with g.as_default():
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
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

    out = utils.deconv(img, model, layer_idx, g=g, sess=sess)
    out = utils.normalize_image(out, per_image=True)
    out = utils.combine_and_fit(out, is_deconv=True, disp_w=1920)
    out = utils.to_255(out)

    cv2.imwrite('{}_deconv_out.png'.format(model.layers[layer_idx].name), out)


if __name__ == '__main__':
    main()
