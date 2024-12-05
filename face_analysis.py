import tensorflow as tf

import os
import time

import matplotlib.pyplot as plt
# from IPython import display

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Use paths relative to DataPreprocessing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "DataPreprocessing")

real_image_path = os.path.join(data_dir, "testing_dataset_splitted/left_split/5582_left_split.jpg")
input_image_path = os.path.join(data_dir, "testing_dataset_splitted/right_split/5582_right_split.jpg")

def loadImage(real_image_path, input_image_path):
    # Load and decode both images
    real_image = tf.io.read_file(real_image_path)
    real_image = tf.image.decode_jpeg(real_image)
    
    input_image = tf.io.read_file(input_image_path)
    input_image = tf.image.decode_jpeg(input_image)
    
    # Resize images to match the expected dimensions
    real_image = tf.image.resize(real_image, [IMG_HEIGHT, IMG_WIDTH])
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH])
    
    # Convert to float32 and normalize to [-1, 1]
    input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1
    real_image = (tf.cast(real_image, tf.float32) / 127.5) - 1
    
    return input_image, real_image

inp, re = loadImage(real_image_path, input_image_path)
# casting to int for matplotlib to show the image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Input Sketch')
plt.imshow(inp * 0.5 + 0.5)  # Denormalize from [-1,1] to [0,1]
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Real Face')
plt.imshow(re * 0.5 + 0.5)  # Denormalize from [-1,1] to [0,1]
plt.axis('off')

plt.show()  # Actually display the plots


##Resizing Images
def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image=tf.stack([input_image, real_image], axis=0)
    cropped_image=tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]





#Convertting pixel values from [0,255] to [-1,1] range
def normalize(input_image, real_image):
    input_image=(input_image/127.5)
    real_image=(real_image/127.5)
    return input_image, real_image

def random_jiter(input_image, real_image):
    # First load the images since we're getting paths
    input_image = tf.io.read_file(input_image)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.cast(input_image, tf.float32)
    
    real_image = tf.io.read_file(real_image)
    real_image = tf.image.decode_jpeg(real_image)
    real_image = tf.cast(real_image, tf.float32)
    
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)
    # cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)
    
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


plt.figure(figsize=(6, 6)) #6*6 inch

for i in range(4):  #4 different augmented images
    #random augementation
    inp, re = random_jiter(input_image_path, real_image_path)
    #normalizing
    inp, re = normalize(inp, re)
    plt.subplot(2, 2, i+1)  # Create a 2x2 grid, place image at position i+1
    plt.imshow(inp * 0.5 + 0.5, cmap='Dark2', interpolation='nearest')  # Use grayscale colormap with black background
    plt.axis('off')

plt.tight_layout()  # Adjust spacing between plots
plt.show()


def load_image_train(image_file):
    input_image, real_image = loadImage(image_file)

