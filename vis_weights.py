import cv2
import numpy as np

import utils


def main():
    from vgg16 import model

    for weight, weight_val in zip(model.weights, model.get_weights()):
        if len(weight_val.shape) == 1:
            continue

        mode = 'fc' if len(weight_val.shape) == 2 else 'conv'

        if mode == 'fc':
            continue

        w = utils.normalize_weights(weight_val, mode)
        w = np.transpose(w, (3, 0, 1, 2))
        disp = utils.combine_and_fit(w, disp_w=800)

        cv2.imshow('disp1', disp)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
