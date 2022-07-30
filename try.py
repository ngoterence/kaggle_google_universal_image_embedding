import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Check the data is correctly formed

BASE_PATH = "data/base_pictures/"
AUGMENTED_PATH = "data/augmented_data/"
HEIGHT, WIDTH = 224, 224

print(len(os.listdir(AUGMENTED_PATH + "images/")))

with open(AUGMENTED_PATH + "labels.json", "r") as read_file:
    labels = json.load(read_file)
print(len(labels))


# ----------------------------------------------------------------------------------------------------

def plot_image_transformations(base_image_id="packaged_accessories_t1_2"):
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10,10))
    indexes = np.arange(20).reshape((4,5))

    for i in range(4):
        for j in range(5):
            img = np.load(AUGMENTED_PATH + "images/{}_{}.npy".format(base_image_id, indexes[i,j]))
            img = img / 255
            axes[i,j].imshow(img)
    plt.show()

    
    
if __name__ == "__main__":
    plot_image_transformations(base_image_id="dishes_t1_2")
