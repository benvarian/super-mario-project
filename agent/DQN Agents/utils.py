from PIL import Image
import numpy as np


def preprocess_state(state):
    '''Resize state and convert to greyscale'''
    image = Image.fromarray(state)
    image = image.resize((88, 80))
    image = image.convert("L") # convert to greyscale
    #     image.show()
    image = np.array(image)

    return image