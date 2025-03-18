#module primarily to store image visualization and results visualization functions

import cv2 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os

IMG_HEIGHT = 256
IMG_WIDTH = 256
cmap_GreenRed = LinearSegmentedColormap.from_list('green-red',["springgreen", "crimson"], N=256) # a_channel
cmap_BlueYellow = LinearSegmentedColormap.from_list('blue-yellow',["blue", "yellow"], N=256) # b_channel
cmap_green_red = cmap_GreenRed #alias names for convenience
cmap_blue_yellow = cmap_BlueYellow #alias names for convenience

def read_image(filename):
  image_bgr = cv2.imread(filename)
  image_bgr = cv2.resize(image_bgr, (IMG_HEIGHT, IMG_WIDTH))
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

  image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
  return image_lab, image_rgb

def load_image_into_disk(filename, save_dir):
  fname = filename.split("/")[-1]

  image_bgr = cv2.imread(filename)
  image_bgr = cv2.resize(image_bgr, (IMG_HEIGHT, IMG_WIDTH))

  image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
  cv2.imwrite(os.path.join(save_dir, fname), image_lab)

def plot_LAB(L_intensity, A_green_red, B_blue_yellow, RGB_image):
  fig, ax = plt.subplots(ncols = 4, nrows = 1, figsize = (16, 4))

  ax[0].imshow(L_intensity, cmap = plt.cm.gray)
  ax[0].set_title("Grayscale", fontsize = 14)
  ax[1].imshow(A_green_red, cmap = cmap_green_red, vmin = 0, vmax = 1)
  ax[1].set_title("Model Color",  fontsize = 14)
  ax[2].imshow(B_blue_yellow,cmap = cmap_blue_yellow, vmin = 0, vmax = 1)
  ax[2].set_title("Anime Color",  fontsize = 14)
  ax[3].imshow(RGB_image)

  for i in range(3): ax[i].axis("off")
  plt.show()

def plot_image(L_intensity, image_rgb_pred, image_rgb_real):
  fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (12, 4))

  ax[0].imshow(L_intensity, cmap = plt.cm.gray)
  ax[0].set_title("Grayscale", fontsize = 14)
  ax[1].imshow(image_rgb_pred)
  ax[1].set_title("Model Color",  fontsize = 14)
  ax[2].imshow(image_rgb_real)
  ax[2].set_title("Anime Color",  fontsize = 14)

  for i in range(3): ax[i].axis("off")
  plt.show()

def plot_RGB(image_rgb):
  red, green, blue = image_rgb[:,:,0],  image_rgb[:,:,1], image_rgb[:,:,2]
  fig, ax = plt.subplots(ncols = 4, nrows = 1, figsize = (16, 4))

  ax[0].imshow(red, cmap = plt.cm.Reds, vmin = 0, vmax = 1)
  ax[0].set_title("Red", fontsize = 14)
  ax[1].imshow(green, cmap = plt.cm.Greens, vmin = 0, vmax = 1)
  ax[1].set_title("Green",  fontsize = 14)
  ax[2].imshow(blue, cmap = plt.cm.Blues,  vmin = 0, vmax = 1)
  ax[2].set_title("Blue",  fontsize = 14)

  ax[3].imshow(image_rgb)
  ax[3].set_title("Image", fontsize = 14)

  for i in range(4): ax[i].axis("off")
  plt.show()

def generate_3D_RGB(image_rgb):
    #note: code is adapted from the Matplotlib tutorial on creating 3D scatter plots 
    fig = plt.figure() #figsize = (10,10))
    ax = fig.add_subplot(projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    #for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    #    xs = randrange(n, 23, 32)
    #    ys = randrange(n, 0, 100)
    #    zs = randrange(n, i, zhigh)
    image_rgb_scaled = image_rgb[:] / 255.0

    ax.scatter(np.ndarray.flatten(image_rgb_scaled[:,:,0]),
            np.ndarray.flatten(image_rgb_scaled[:,:,1]),
            np.ndarray.flatten(image_rgb_scaled[:,:,2]),
            marker='o',
            c = [color for row in image_rgb_scaled for color in row])

    ax.set_xlabel('R (Red)')
    ax.set_ylabel('G (Green)')
    ax.set_zlabel('B (Blue)')

    plt.show()

def generate_3D_CIELAB(image_lab, image_rgb):
    #note: code is adapted from the Matplotlib tutorial on creating 3D scatter plots 

    fig = plt.figure() #figsize = (10,10))
    ax = fig.add_subplot(projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    #for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    #    xs = randrange(n, 23, 32)
    #    ys = randrange(n, 0, 100)
    #    zs = randrange(n, zlow, zhigh)
    image_lab_scaled = image_lab[:]

    ax.scatter(np.ndarray.flatten(image_lab_scaled[:,:,0]),
            np.ndarray.flatten(image_lab_scaled[:,:,1]) - 0.5,
            np.ndarray.flatten(image_lab_scaled[:,:,2]) - 0.5,
            marker='o',
            c = [color for row in image_rgb /255.0 for color in row])

    ax.set_xlabel('L* (Lightness)')
    ax.set_ylabel('A* (Green-Red)')
    ax.set_zlabel('B* (Blue-Yellow)')

    plt.show()

