import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import pandas as pd
from PIL import Image
import cv2, glob
import sklearn
from skimage import morphology
from itertools import permutations

"""
Crops your images down to an ROI based on some dimension I 
tentatively found (and seem to hold).

image_path is just going to be the path to the raw screen shots.
"""
def crop(images_path):
    images = []
    dir = os.listdir(images_path)
    # These were just found by hand on the first 50 samples and then confirmed on
    # the larger data set
    x1 = 428
    y1 = 154
    x2 = 942
    y2 = 292
    ROI_num = 1
    crop_path = "C:/Users/Jackson/Desktop/ms_data/cropped_images"

    for path in dir:
        reformatted_path = os.path.abspath(images_path + "/" + path)
        images = np.append(images, reformatted_path)

    for image in images:
        # You can visualize the box by uncommenting some of the stuff below
        #rect = patches.Rectangle(xy = (x1, y2), width= (x2-x1), height= (y1 - y2), edgecolor= "r", linewidth= 2,fill = False)
        img = cv2.imread(image, 1)
        ROI = img[y1:y2, x1:x2]
        cv2.imwrite("C:/Users/Jackson/Desktop/ms_data/cropped_images/ROI_{}.png".format(ROI_num), ROI)
        ROI_num += 1
        # Visualizing the ROI
        #im = plt.imread(image)
        #ax = plt.gca()
        #fig = plt.gcf()
        #plot = ax.imshow(ROI)
        #fig.canvas.mpl_connect('key_press_event', input)
        #ax.add_patch(rect)
        #plt.show()

"""
Will apply the four image transformations and resize them.

images_path will be the path of the cropped images create from the crop() call.
"""
def preprocess(images_path):
    images = []
    dir = os.listdir(images_path)
    counter = 1

    for path in dir:
        reformatted_path = os.path.abspath(images_path + "/" + path)
        images = np.append(images, reformatted_path)

    for image in images:
        img = cv2.imread(image).copy()
        img = cv2.GaussianBlur(img, (3,3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        coefficients = (0.0445, 0.6568, 0.2987)
        img = cv2.transform(img, np.array(coefficients).reshape((1, 3)))
        # Trying without morphology at first
        img = morphology.remove_small_objects(img, min_size=35, connectivity=5)
        resize = cv2.resize(img, (int(img.shape[1] * .5), int(img.shape[0] * .5)))
        # print(resize.shape)
        cv2.imwrite("C:/Users/Jackson/Desktop/ms_data/processed_images/pp_{}.png".format(counter), resize)
        counter += 1


"""
Will create the data set as an np.array.

images_path will be the directory that points to the processed images created from
preprocess().
"""
def make_data_set(images_path):
    images = []
    dir = os.listdir(images_path)
    data_set = []

    for path in dir:
        reformatted_path = os.path.abspath(images_path + "/" + path)
        images = np.append(images, reformatted_path)


    for image in images:
        im = cv2.imread(image)
        data_set = np.append(data_set, im)

    data_set = data_set.reshape((-1, 69, 257, 3))
    print(data_set.shape)

    return data_set

"""
Converts your raw inputs from the labeling process to numerical labels corresponding
to the mapping 1: "up", 2: "down", 3: "left", 4: "right".

string_labels is the csv file holding the raw string input labels.
"""
def convert_to_numeric(string_labels):
    numeric_labels = np.zeros(shape=string_labels.shape, dtype=np.int)

    for i in range(string_labels.shape[0]):
        for j in range(string_labels.shape[1]):
            if string_labels[i,j] == 'up':
                numeric_labels[i,j] = 1
            elif string_labels[i,j] == 'down':
                numeric_labels[i,j] = 2
            elif string_labels[i,j] == 'left':
                numeric_labels[i,j] = 3
            elif string_labels[i,j] == 'right':
                numeric_labels[i,j] = 4

    return numeric_labels


def main():
    file_path = "C:/Users/Jackson/Desktop/ms_data/raw_images"
    # Only need to run once to crop all images
    console = input("What would you like to do: \n - 1: Crop \n - 2: Transform \n - 3: Make Data Set \n Input: ")

    if console == '1':
        # The directory where the original screen shots are.
        crop(file_path)

    elif console == '2':
        # The directory where your cropped images will be.
        preprocess("C:/Users/Jackson/Desktop/ms_data/cropped_images")

    elif console == '3':
        labels = pd.read_csv("labels.csv", header = None)
        labels = labels.values
        # Numeric labels are (1000, 4)
        numeric_labels = convert_to_numeric(labels)
        print(numeric_labels.shape)
        np.save("C:/Users/Jackson/Desktop/ms_data/numeric_labels", numeric_labels)
        # Full data set is (1000, 69, 257, 3)
        data = make_data_set("C:/Users/Jackson/Desktop/ms_data/processed_images")
        np.save("C:/Users/Jackson/Desktop/ms_data/data", data)

if __name__ == '__main__':
    main()
