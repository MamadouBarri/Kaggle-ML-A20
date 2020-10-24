from Src import data_extraction
import numpy as np
import matplotlib.pyplot as plt
from Src.neural_network import neural_network

LEARNING_RATE = 0.05
SAVING = False

def main():

    # Paths
    train_path = "data_train.npz/data_train.npz"
    test_path = "data_test.npz/data_test.npz"
    saving_path_data = 'saves/formatted_data.npy'
    saving_path_labels = 'saves/formatted_labels.npy'

    data = 0
    labels = 0
    # Extracting and saving data if needed
    if SAVING:
        # Extracting data
        (data, labels) = data_extraction.extract_images(train_path)
        
        np.save(saving_path_data, data)
        np.save(saving_path_labels, labels)
    else:
        data = np.load(saving_path_data)
        labels = np.load(saving_path_labels)

    # Taking a percentage of the training set for validation
    (train_x, val_x, train_y, val_y) = data_extraction.split_data(data, labels, 0.8)

    # Creating neural network
    nn = neural_network(learning_rate=0.01, hidden_layers_sizes=(64, 64, 32, 32), data=train_x, labels=train_y)
    
    print('Before : ', nn.measure_classification(val_x, val_y))

    # Train model
    for i in range(0, 100):
        data_extraction.shuffle(train_x, train_y)
        (current_data, current_labels) = data_extraction.subset(train_x, train_y, 0.2)
        nn.train(current_data, current_labels)


    print('After : ', nn.measure_classification(val_x, val_y))


if __name__ == "__main__":
    main()



