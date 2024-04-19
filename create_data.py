import string

import numpy as np
from freetype import *
from matplotlib import pyplot as plt

font_file = "data/NotoSansMono-VariableFont_wdth,wght.ttf"
chars = list(string.ascii_uppercase)

print(chars)

for c, char in enumerate(chars):
    face = Face(font_file)
    face.set_char_size(96 * 64)

    face.load_char(char)

    outline = face.glyph.outline
    points = outline.points

    points = np.array(points, dtype=np.float32)

    max_y = max([point[1] for point in points])

    points[:, 1] = max_y - points[:, 1]

    # Normalize the points
    points /= np.max(points)

    np.save(f'data/{char}.npy', points)

    """
    # Separate x and y coordinates
    x, y = zip(*points)
    
    # Plot the shape
    plt.plot(x, y)

    # Add text annotations to each point
    for i, (xi, yi) in enumerate(outline.points):
        plt.text(xi, yi, f'({i})', ha='right', va='bottom')

    # Show the plot (optional)
    plt.show()
    """
