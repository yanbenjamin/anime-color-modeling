import os
import glob 
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import os 

from utils import load_image_into_disk

def get_all_images(dir):
    images_list = []
    for file_extension in [".jpg", ".jpeg", ".png"]:
        images_list += glob.glob(os.path.join(dir, f"*{file_extension}"))
    return images_list

def split_train_test(images, TRAIN_FRACTION):
    image_filenames = images[:]
    #bucket the images into the training and testing sets 
    np.random.shuffle(image_filenames)
    TRAIN_FRACTION = 0.8
    train_images = image_filenames[:int(len(image_filenames) * TRAIN_FRACTION)]
    test_images = image_filenames[int(len(image_filenames) * TRAIN_FRACTION):]
    return train_images, test_images

if __name__ == "__main__":
    #first, you'll want to download the dataset at https://www.kaggle.com/datasets/weiwangk/japanese-anime-scenes, unzip it, 
    #and place the folder of JPEG images into the same directory as this file 

    IMAGES_DIR = "./Japanese-Anime-Scenes/"
    TRAIN_FRACTION = 0.8 # 80% of the data / images goes toward training, rest toward testing 
    image_list = get_all_images(IMAGES_DIR)

    train_images, test_images = split_train_test(image_list, TRAIN_FRACTION)
    num_train = len(train_images)
    num_test = len(test_images)
    print(f"Train Images (n = {num_train}), Test Images (n = {num_test})")

    #you'll also want to run mkdir train and mkdir test before running the following
    #which reads and saves the images (in the LAB color space) into those new folders

    print("Processing and Saving the Training Images")
    for img in tqdm(train_images):
        load_image_into_disk(img, "./train")

    print("Processing and Saving the Testing Images")
    for img in tqdm(test_images):
        load_image_into_disk(img, "./test")
    





    

