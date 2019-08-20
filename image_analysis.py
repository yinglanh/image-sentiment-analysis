import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "image"
CATEGORIES = ["pos", "neg"]

for category in  CATEGORIES:
    path = os.path.join(DATADIR, category) #path to pos or neg dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break

# %%

print(img_array.shape)

# %%

IMG_SIZE = 150

new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap="gray")
plt.show()

# %%
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to pos or neg dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass # or os error

create_training_data()

# %%
print(len(training_data))

# %%
import random
random.shuffle(training_data)

# %%
for sample in training_data[:10]:
    print(sample[1])

# %%
X = []  # feature set
y = []  # labels

# %%
for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# %%
import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# %%
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

# %%
X[1]







