import numpy as np
from Src.preprocessing import interpolatePixels, IMAGE_SIZE

def load_data(path):
    return np.load(path)


def interpolate_images(images):
    for image in range(len(images)):
        interpolatePixels(images[image])


def flatten_images(images):

    flattened_images = []
    for image in range(len(images)):

        # Reshape to a onedimensional vector
        flattened_images.append(np.reshape(
            images[image], (IMAGE_SIZE * IMAGE_SIZE * 3, 1)))

    return flattened_images

# Extracts images from [train_path] and returns the images and the labels in the right format (50000x3072)


def extract_images(train_path):
    train_images = load_data(train_path)

    # Get images' data and labels
    data = train_images['data'][0:100]
    labels = np.array(train_images['labels'])

    # Remove black squares from image
    interpolate_images(data)

    # Flatten every images
    data = np.array(flatten_images(data))
    return (data, labels)

'''
train_path = "polyai-ml-a20/data_train.npz"
test_path = "polyai-ml-a20/data_test.npz"

images = extract_images(train_path)
data = images[0]
labels = images[0]
print(data.shape)
'''