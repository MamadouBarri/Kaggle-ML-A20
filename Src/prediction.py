import numpy as np

classes = [ 'aquatic mammals', 'fish', 'flowers',
            'food containers', 'fruit and vegetables', 'household electrical devices',
            'household furniture', 'insects', 'large carnivores', 'large man-made outdoor things',
            'large natural outdoor scenes', 'large omnivores and herbivores', 'medium mammals',
            'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
            'trees', 'vehicles 1', 'vehicles 2']

def output_from_label(label: str):
    position = classes.index(label)
    output = np.zeros(len(classes))
    output[position] = 1
    return output
