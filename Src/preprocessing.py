import numpy as np  # linear algebra


IMAGE_SIZE = 32

# Returns the rgb values of a pixel in an image


def getRGB(image, position):
    return image[position[0], position[1]]

# Checks whether the current position is in the image


def isPixelPositionValid(width, height, position):
    return not(position[0] < 0 or position[0] >= width or position[1] < 0 or position[1] >= height)

# Checks whether the value of r, g and b is [x] in an array


def areAllChannels(arr, x):
    return arr[0] == x and arr[1] == x and arr[2] == x

# Checks whether the current position is in the image and that the pixel is not black


def isPixelValid(image, width, height, position):
    return isPixelPositionValid(width, height, position) and not(areAllChannels(getRGB(image, position), 0))

# Returns the neighbouring pixels of a pixel in a image


def getNeighbourPixels(image, position):

    pixels = []

    left = [position[0] - 1, position[1]]
    right = [position[0] + 1, position[1]]
    top = [position[0], position[1] - 1]
    bottom = [position[0], position[1] + 1]

    if (isPixelValid(image, IMAGE_SIZE, IMAGE_SIZE, left)):
        pixels.append(left)
    if (isPixelValid(image, IMAGE_SIZE, IMAGE_SIZE, right)):
        pixels.append(right)
    if (isPixelValid(image, IMAGE_SIZE, IMAGE_SIZE, top)):
        pixels.append(top)
    if (isPixelValid(image, IMAGE_SIZE, IMAGE_SIZE, bottom)):
        pixels.append(bottom)

    return pixels


# Gets the mean value of the surrounding pixels
def interpolatePixel(image, position):

    neighbours = getNeighbourPixels(image, position)

    mean_values = np.zeros((len(neighbours), 3))
    if len(mean_values) > 0:
        for n in range(len(neighbours)):
            mean_values[n] += getRGB(image, neighbours[n])

    mean_values = np.sum(mean_values, axis=0)
    if len(neighbours) != 0:
        mean_values = np.divide(mean_values, len(neighbours))

    return mean_values


# Interpolates pixels of images to fill the black pixels
def interpolatePixels(image):
    for x in range(IMAGE_SIZE):
        for y in range(IMAGE_SIZE):

            black = areAllChannels(image[x, y], 0)

            if black:
                image[x, y] = interpolatePixel(image, [x, y])
