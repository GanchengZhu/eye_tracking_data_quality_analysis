# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np

# Constants
PHONE_SCREEN_WIDTH_PIXELS = 1080
PHONE_SCREEN_WIDTH_CM = 7.4
PHONE_SCREEN_HEIGHT_PIXELS = 2249
PHONE_SCREEN_HEIGHT_CM = 15.34293
EYELINK_SCALE_FACTOR_X_CM = PHONE_SCREEN_WIDTH_CM / 332  # 7.4 / 332
EYELINK_SCALE_FACTOR_Y_CM = PHONE_SCREEN_HEIGHT_CM / 692  # 15.34293 / 692
EYELINK_INVALID_VALUE = -32768.0
DATA_LOSS_THRESHOLD = 0.15  # 15% data loss
WINDOW_SIZE_MS = 175
EYELINK_SAMPLE_RATE = 500  # Hz
EYELINK_SKIP_IDS = {6, 23, 31, 32}

AVERAGE_DISTANCE = 36.58701270997024


def get_phone_size():
    deg_height = np.rad2deg(np.arctan(PHONE_SCREEN_HEIGHT_CM / 2 / AVERAGE_DISTANCE)) * 2
    deg_width = np.rad2deg(np.arctan(PHONE_SCREEN_WIDTH_CM / 2 / AVERAGE_DISTANCE)) * 2
    print(f'width * height = {deg_width} * {deg_height} (in centimeters)')


if __name__ == '__main__':
    get_phone_size()
