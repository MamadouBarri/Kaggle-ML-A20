
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import preprocessing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load files
data_train_file = np.load("polyai-ml-a20/data_train.npz")
data_test_file = np.load("polyai-ml-a20/data_test.npz")

print(data_train_file.files)
print(data_test_file.files)

images_train = data_train_file['data']
labels_train = data_train_file['labels']
labels_metadata = data_train_file['metadata'].astype(str)
image0 = images_train[65]
label0 = labels_train[65]

plt.imshow(image0)
plt.show()
preprocessing.interpolatePixels(image0)
plt.imshow(image0)
plt.show()







