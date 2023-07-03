import os
import random
import string

import numpy as np
import torch
from PIL import Image
from freetype import *
from torchvision import transforms
from tqdm import tqdm
import cv2

from generator.vae import VQVAE
from utils import draw_three


def lindexsplit(some_list, *args):
    # Checks to see if any extra arguments were passed. If so,
    # prepend the 0th index and append the final index of the
    # passed list. This saves from having to check for the beginning
    # and end of args in the for-loop. Also, increment each value in
    # args to get the desired behavior.
    if args:
        args = (0,) + tuple(data+1 for data in args)

    # For a little more brevity, here is the list comprehension of the following
    # statements:
    #    return [some_list[start:end] for start, end in zip(args, args[1:])]
    my_list = []
    for start, end in zip(args, args[1:]):
        my_list.append(some_list[start:end])
    return my_list


path = "data/"
n_fonts = 0
skipped_fonts = 0
# chars = list(string.ascii_uppercase)
chars = ["A"]
fonts = []
labels = []
lengths = []
styles = []
# images = []

vae = VQVAE()
vae.load_state_dict(torch.load("vqvae.pth"))

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


for f, filename in enumerate(tqdm(os.listdir(path))):
    filename = filename.lower()
    fontpath = path + filename

    if filename.endswith(".ttf") or filename.endswith(".otf") or filename.endswith(".pfb") or filename.endswith(".pfa"):
        for c, char in enumerate(chars):
            face = Face(fontpath)
            face.set_char_size(96 * 64)

            face.load_char(char)

            """
            glyph = face.glyph.get_glyph()

            blyph = glyph.to_bitmap(FT_RENDER_MODE_NORMAL, Vector(0, 0), True)
            bitmap = blyph.bitmap

            width, rows, pitch = bitmap.width, bitmap.rows, bitmap.pitch
            top, left = blyph.top, blyph.left

            large_side = max(width, rows)
            img = np.zeros((large_side, large_side))

            yoff = round((large_side - rows) / 2)
            xoff = round((large_side - width) / 2)

            over = np.array(bitmap.buffer, dtype='ubyte').reshape(rows, width)

            img[yoff:yoff + rows, xoff:xoff + width] = over

            img = Image.fromarray(np.uint8(img)).convert('RGB')
            img = img.resize((224, 224))

            imgname = filename.split(".")[0]
            img.save(f"images/{imgname}.png")

            """

            outline = face.glyph.outline

            y = [t[1] for t in outline.points]

            strokes = lindexsplit(outline.points, *outline.contours)

            outline_three = []
            current_x, current_y = 0, 0
            for stroke in strokes:
                stroke.append(stroke[0])
                for p, point in enumerate(stroke):
                    if p != 0 or stroke != strokes[0]:
                        new_x = (point[0] - current_x)
                        new_y = ((max(y) - point[1]) - current_y)

                        if p == (len(stroke) - 1):
                            outline_three.append([new_x, new_y, 1])
                        else:
                            outline_three.append([new_x, new_y, 0])

                    current_x = point[0]
                    current_y = (max(y) - point[1])

            outline_three = np.array(outline_three, dtype=np.float64)

            if len(outline_three) != 0:
                fonts.append(outline_three)
                n_fonts += 1
                lengths.append(len(outline_three))
                labels.append(c)

                imgname = filename.split(".")[0]
                img = Image.open(f"images/{imgname}.png")
                img = transform(img)

                quant_t, quant_b, diff, _, _ = vae.encode(img)

                print(quant_t.shape, quant_b.shape)
            else:
                print("No points:", filename)
                skipped_fonts += 1

    else:
        skipped_fonts += 1
        print("Skipped file:", filename)
        continue

    break


"""
print("Loaded fonts:", n_fonts, "Skipped Fonts:", skipped_fonts)

fonts = np.array(fonts, dtype=object)
print(fonts.shape)
labels = np.array(labels)
np.savez('fonts.npz', data=fonts, labels=labels, styles=styles)

# mean = np.mean(lengths)
# std = np.std(lengths)

# print(mean, std)


outline_points = torch.tensor(random.choice(fonts))

img = draw_three(outline_points, random_color=True)

img.save("tests/test_sketch.png")
"""

