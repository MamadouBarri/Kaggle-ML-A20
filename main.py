from Src import data_extraction
import matplotlib.pyplot as plt
from Src import neural_network

ITERATION = 1000
LEARNING_RATE = 0.05

def main():

    # Paths
    train_path = "polyai-ml-a20/data_train.npz"
    test_path = "polyai-ml-a20/data_test.npz"

    (data, labels) = data_extraction.extract_images(train_path)
    mlp = neural_network.build_mlp()
    
    for i in range(ITERATION):

        # Train
        

if __name__ == "__main__":
    main()



