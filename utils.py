import math

import cv2
import numpy as np
from keras import backend as K


def get_layers(x, model, out_idx):
    f = K.function([model.layers[0].input],
                   [model.layers[out_idx].output])
    out = f([x])[0]
    return out


def normalize_image(img, gamma=1.0):
    min_img = img.min()
    max_img = img.max()
    return (img - min_img) / (max_img - min_img)


def normalize_weights(weights, mode, gamma=1.0):
    if mode == 'conv':
        if weights.shape[2] == 3:  # (h, w, c, n_filter)
            new_weights = normalize_image(weights)
        else:
            new_weights = np.zeros(weights.shape)
            for i in range(weights.shape[3]):
                new_weights[..., i] = normalize_image(weights[..., i], gamma)
        return new_weights


def rgb_to_bgr(img):
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        return img[:, :, ::-1]
    else:
        return img[:, :, :, ::-1]


def combine_and_fit(data, gap=1, is_conv=False, is_fc=False, disp_w=None):
    if len(data.shape) == 4:
        h, w = data.shape[1:3]
        color = data.shape[-1] == 3
    else:
        h, w = 1, 1
        color = False

    total = len(data)

    if color:
        factor = 10
        n_col = int(math.ceil(math.sqrt(total)))
        n_col_gap = n_col - 1
        width = int(n_col * w * factor + n_col_gap * gap)
        height = int(n_col * h * factor + n_col_gap * gap)
        y_jump = int(h * factor + gap)
        x_jump = int(w * factor + gap)
        img = np.zeros((height, width, 3), dtype=np.float32)
        img += 0.1
        i = 0
        for y in range(n_col):
            y = y * y_jump
            for x in range(n_col):
                x = x * x_jump
                img[y:y + h * factor, x:x + w * factor] = cv2.resize(data[i], None, fx=factor, fy=factor,
                                                                     interpolation=cv2.INTER_AREA)
                i += 1
        return img
    else:
        if is_conv or is_fc:
            n_row = int(math.ceil(math.sqrt(total)))
            n_col = n_row
        else:
            n_row = total
            n_col = data.shape[-1]


        n_col_gap = n_col - 1
        n_row_gap = n_row - 1

        factor = (disp_w - n_col_gap * gap) / float(n_col * w)
        width = int(n_col * w * factor + n_col_gap * gap)
        height = int(n_row * h * factor + n_row_gap * gap)
        y_jump = int(h * factor + gap)
        x_jump = int(w * factor + gap)
        new_w = int(w * factor)
        new_h = int(h * factor)

        img = np.zeros((height, width), dtype=np.float32)
        img += 0.2

        i = 0
        for y in range(n_row):
            y = y * y_jump
            j = 0
            for x in range(n_col):
                x = x * x_jump

                if i >= total:
                    break

                if is_conv:
                    d = data[i, :, :]
                elif is_fc:
                    d = data[i]
                else:
                    d = data[i, :, :, j]

                to_y = y + int(h * factor)
                to_x = x + int(w * factor)
                img[y:to_y, x:to_x] = cv2.resize(d, (new_w, new_h),
                                                 interpolation=cv2.INTER_AREA)

                if is_conv or is_fc:
                    i += 1
                else:
                    j += 1

            if not is_conv and not is_fc:
                i += 1
        return img
