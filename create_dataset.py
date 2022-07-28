import cv2
from scipy.ndimage import zoom
import numpy as np
from PIL import ImageEnhance
import PIL
import os
from tqdm import tqdm
import json

BASE_PATH = "data/base_pictures/"
AUGMENTED_PATH = "data/augmented_data/"
HEIGHT, WIDTH = 224, 224

def list_all_base_pictures_paths():
    """
    @ inputs:
    * None

    @ output:
    * img_paths (list of str): the paths to every base images (in data/base_pictures/ folder)
    """
    img_paths = []
    for dir in os.listdir(BASE_PATH):
        category_path = os.path.join(BASE_PATH, dir)
        for file in os.listdir(category_path):
            new_img_path = os.path.join(category_path, file)
            img_paths.append(new_img_path)
    return img_paths

def list_instance_pictures_paths(category="landmarks", id="t1"):
    """
    @ inputs:
    * category (str): category of the instance to retrieve
    * id (str): id of the instance to retrieve

    @ output:
    * paths (list of str): the paths to every images of the given id instance (in data/base_pictures/category folder)
    """
    paths = []
    assert (category in os.listdir(BASE_PATH))
    category_path = os.path.join(BASE_PATH, category)
    for file in os.listdir(category_path):
        if file.startswith(id):
            new_path = os.path.join(category_path, file)
            paths.append(new_path)
            print(new_path)
    return paths

def create_augmented_images(img_path, new_height=224, new_width=224):
    """
    @ inputs:
    * img_path (str): path of the image to be augmented
    * new_height (int): height of the resized images
    * new_width (int): width of the resized images

    @ output:
    versions (np.array of shape (20,new_height,new_width,3)): array containing 20 different versions of the original image 
    """
    def resize_img(img, new_height=224, new_width=224):
        h,w = img.shape[:2]
        img_resized = zoom(img, (new_height/h, new_width/w, 1))
        return img_resized

    versions = np.zeros((20, new_height, new_width, 3))
    img_ref = np.array(PIL.Image.open(img_path))
    # filtres seuls
    img_resized = resize_img(img_ref, new_height, new_width)
    versions[0] = img_resized

    converter = ImageEnhance.Brightness(PIL.Image.open(img_path))
    img_dark = converter.enhance(0.5)
    img_dark = resize_img(np.array(img_dark), new_height, new_width)
    versions[1] = img_dark

    img_light = converter.enhance(1.5)
    img_light = resize_img(np.array(img_light), new_height, new_width)
    versions[2] = img_light

    img_blured = cv2.GaussianBlur(img_resized, (3,3), 2, 2)
    img_blured = resize_img(np.array(img_blured), new_height, new_width)
    versions[3] = img_blured

    converter = ImageEnhance.Contrast(PIL.Image.open(img_path))
    img_contrast = converter.enhance(3)
    img_contrast = resize_img(np.array(img_contrast), new_height, new_width)
    versions[4] = img_contrast
    # dark + rotations
    versions[5] = cv2.rotate(img_dark, cv2.ROTATE_90_CLOCKWISE)
    versions[6] = cv2.rotate(img_dark, cv2.ROTATE_90_COUNTERCLOCKWISE)
    versions[7] = cv2.rotate(img_dark, cv2.ROTATE_180)
    # light + rotations
    versions[8] = cv2.rotate(img_light, cv2.ROTATE_90_CLOCKWISE)
    versions[9] = cv2.rotate(img_light, cv2.ROTATE_90_COUNTERCLOCKWISE)
    versions[10] = cv2.rotate(img_light, cv2.ROTATE_180)
    # blured + rotations
    versions[11] = cv2.rotate(img_blured, cv2.ROTATE_90_CLOCKWISE)
    versions[12] = cv2.rotate(img_blured, cv2.ROTATE_90_COUNTERCLOCKWISE)
    versions[13] = cv2.rotate(img_blured, cv2.ROTATE_180)
    # contrast + rotations
    versions[14] = cv2.rotate(img_contrast, cv2.ROTATE_90_CLOCKWISE)
    versions[15] = cv2.rotate(img_contrast, cv2.ROTATE_90_COUNTERCLOCKWISE)
    versions[16] = cv2.rotate(img_contrast, cv2.ROTATE_180)
    # rotations seules
    versions[17] = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)
    versions[18] = cv2.rotate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
    versions[19] = cv2.rotate(img_resized, cv2.ROTATE_180)
    return versions

# Necessary to build the labels
def group_paths_by_instance(paths):
    """
    @ inputs:
    * paths (list of str): list of paths of the form "data/base_pictures/..."

    @ outputs:
    * res (list of lists of str): described by the name of the function
    """
    ids = np.unique([path[:path.rindex("_")] for path in paths]) # ids take into account the categories of the pictures
    res = [[path for path in paths if path.startswith(id)] for id in ids]
    return res


def add_instance_to_dataset(category="landmarks", id_instance="t1"):
    paths = list_instance_pictures_paths(category, id_instance)

    with open(AUGMENTED_PATH + "labels.json", "r") as read_file:
        labels = json.load(read_file)
    
    # check if the instance is already stored
    print("Check whether the instance is already stored")
    pattern = category + "_" + id_instance
    stored_ids = list(labels.keys())
    for stored_id in stored_ids:
        if stored_id.startswith(pattern):
            print("This instance is already stored!")
            return 1

    print("Start the updates")
    new_label = int(np.max(list(labels.values())) + 1)
    ids = [path[len(BASE_PATH):].replace("/", "_")[:-4] for path in paths] #  category/[v|t][num_instance]_[num_picture_instance] (ID of each base image)

    # update labels.json
    for id in ids:
        for i in range(20): # each base image is declined into 20 versions via create_augmented_images
            labels[id + "_" + str(i)] = new_label
            
    with open(AUGMENTED_PATH + "labels.json", 'w') as f:
        json.dump(labels, f)
    print(len(labels))

    # add images 
    print("Store images")
    pbar = tqdm(range(len(paths)))
    for id, path in zip(ids,paths):
        new_imgs = create_augmented_images(path, new_height=HEIGHT, new_width=WIDTH)
        for i in range(len(new_imgs)): # store each version of the current base image
            with open(AUGMENTED_PATH + "images/" + id + "_" + str(i) + ".npy" , "wb") as file:
                np.save(file, new_imgs[i])
        pbar.update(n=1)



# ------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """ ADD THE AUGMENTED IMAGES OF A NEW INSTANCE TO THE DATASET """
    add_instance_to_dataset(category="others", id_instance="t1")


    

    









