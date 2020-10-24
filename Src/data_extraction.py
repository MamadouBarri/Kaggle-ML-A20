import numpy as np
from sklearn.model_selection import train_test_split
from Src.preprocessing import interpolatePixels, IMAGE_SIZE

def load_data(path):
    return np.load(path)


def interpolate_images(images):
    for image in range(len(images)):
        if image % 1000 == 0:
            print(image)
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
    data = train_images['data']
    labels = np.array(train_images['labels'])

    # Remove black squares from image
    interpolate_images(data)

    # Flatten every images
    data = np.array(flatten_images(data)).squeeze()
    return (data, labels)

def subset(X, Y, percentage):
    return (X[0:int(percentage * len(X))], Y[0:int(percentage * len(X))])

def shuffle(X, Y):
    np.random.shuffle(X)
    np.random.shuffle(Y)
    
# Returns the data splitted with the format (first_split_x, second_split_x, first_split_y, second_split_y)
def split_data(X, Y, percentage):

    if percentage < 0 or percentage > 1:
        raise Exception('Percentage has to be between 0 and 1')

    return (X[0:int(percentage * len(X))], X[int(percentage * len(X)):], Y[0:int(percentage * len(X))], Y[int(percentage * len(X)):])

'''
train_path = "polyai-ml-a20/data_train.npz"
test_path = "polyai-ml-a20/data_test.npz"

images = extract_images(train_path)
data = images[0]
labels = images[0]
print(data.shape)
'''