import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image


"""
Will take your raw images directory, iterate through each image and you will have
to manually input the labels. Once you have entered the correct label you 
will press "e" to move on to the next, if you mess up you can re-input the sequence
by pressing "r". "q" will exit you from the iteration and save your progress.

images_path is the path to the raw images directory.
"""
def get_labels(images_path):
    unlabeled_images = []
    image_labels = []
    input.directions = []
    dir = os.listdir(images_path)
    num_labeled = 0
    input.break_early = False

    for path in dir:
        reformatted_path = os.path.abspath(images_path + "/" + path)
        unlabeled_images = np.append(unlabeled_images, reformatted_path)

    for image in unlabeled_images:
        print("-" * 100)
        print("Image ", num_labeled + 1, "(", unlabeled_images.shape[0] - num_labeled, " remaining): ")
        print("-"*100)
        im = plt.imread(image)
        ax = plt.gca()
        fig = plt.gcf()
        plot = ax.imshow(im)
        fig.canvas.mpl_connect('key_press_event', input)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

        if input.break_early:
            print()
            print("=" * 100)
            print(num_labeled, " Images Labeled. ")
            print("=" * 100)
            break

        else:
            image_labels = np.append(image_labels, input.directions)
            # Then just reset input.directions array
            input.directions = []
            num_labeled += 1

    image_labels = np.reshape(image_labels, newshape=(num_labeled, 4))
    #Saves current progress
    np.savetxt("C:/Users/Jackson/Desktop/ms_data/labels_new.csv", image_labels, fmt='%s', delimiter=",")
    print("="*100)
    print("All Images Labeled. ")
    print("=" * 100)

def input(event):
    if event.key not in ['left', 'right', 'up', 'down', 'r', 'q', 'e']:
        print('Invalid input.')

    else:
        # Take in arrow key inputs
        if event.key in ['left', 'right', 'up', 'down']:
            input.directions = np.append(input.directions, event.key)
            print("Input: ", event.key)

        # Retry entry
        elif event.key == 'r':
            print()
            print("Re-Enter Full Sequence: ")
            print()
            input.directions = []

        elif event.key == 'q':
            print()
            print("Exiting...")
            plt.close()
            input.break_early = True
            return

        elif event.key == 'e':
            if len(input.directions) < 4 or len(input.directions) > 4:
                print()
                print("You have entered more or less than 4 directions!")
                print()
            else:
                print("Input Directions: ", input.directions)
                print()
                plt.close()
                return input.directions


def main():
    file_path = "C:/Users/Jackson/Desktop/ms_data/raw_images"
    get_labels(file_path)


if __name__ == "__main__":
    main()
