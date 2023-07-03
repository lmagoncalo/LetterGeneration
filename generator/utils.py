import random

import cv2
import numpy as np
from PIL import Image


def canvas_size_google(sketch):
    vertical_sum = np.cumsum(sketch[1:], axis=0)
    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)
    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]
    start_y = -ymin - sketch[0][1]
    return [int(start_x), int(start_y), int(h), int(w)]


def scale_sketch(sketch, size=(448, 448)):
    [_, _, h, w] = canvas_size_google(sketch)

    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=np.float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=np.float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=np.float)
    return sketch_rescale.astype("int16")


def draw_three(sketch, random_color=True, img_size=512):
    thickness = int(img_size * 0.01)

    sketch = sketch.cpu().detach().numpy()

    sketch = scale_sketch(sketch, (img_size, img_size))  # scale the sketch.
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += thickness + 1
    start_y += thickness + 1
    canvas = np.ones((max(h, w) + 3 * (thickness + 1), max(h, w) + 3 * (thickness + 1), 3), dtype='uint8') * 255
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        color = (0, 0, 0)
    pen_now = np.array([start_x, start_y])
    first_zero = False
    for s, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        # cv2.putText(canvas, str(s), tuple(pen_now), cv2.FONT_HERSHEY_PLAIN, 1, 255)
        if int(state) == 1:  # next stroke
            first_zero = True
            if random_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = (0, 0, 0)
        pen_now += delta_x_y

    canvas = cv2.resize(canvas, (img_size, img_size))

    # You may need to convert the color.
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    return im_pil
