import tensorflow as tf

import os
import math
import numpy as np
from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from einops import rearrange, reduce

crop_size = 256
upscale_factor = 4
input_size = crop_size // upscale_factor

def process_input(input, input_size, upscale_factor):
    """crop the image, retrieve the y channel (luninance), and resize it with the area method"""
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
    """crop the image and retrieve the y channel"""
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


def load_model(model_path):
    return keras.models.load_model(model_path)

def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )
    
def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img

def plot_results(img, prefix, title):
    """Plot the result with zoom-in area."""
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    # Create a new figure with a default 111 subplot.
    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(title)
    # zoom-factor: 2.0, location: upper-left
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(img_array[::-1], origin="lower")

    # Specify the limits.
    x1, x2, y1, y2 = 200, 300, 100, 200
    # Apply the x-limits.
    axins.set_xlim(x1, x2)
    # Apply the y-limits.
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
    plt.savefig(str(prefix) + "-" + title + ".png")


def run_model(test_path):
    
    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0
    
    test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpg")
    ])
    for index, test_img_path in enumerate(test_img_paths):
        img = load_img(test_img_path)
        lowres_input = get_lowres_image(img, upscale_factor)
        w = lowres_input.size[0] * upscale_factor
        h = lowres_input.size[1] * upscale_factor
        highres_img = img.resize((w, h))
        prediction = upscale_image(model, lowres_input)
        lowres_img = lowres_input.resize((w, h))
        lowres_img_arr = img_to_array(lowres_img)
        highres_img_arr = img_to_array(highres_img)
        predict_img_arr = img_to_array(prediction)
        bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
        test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)
        
        total_bicubic_psnr += bicubic_psnr
        total_test_psnr += test_psnr

        print(
            "PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr
        )
        print("PSNR of predict and high resolution is %.4f" % test_psnr)
        
        if index % 5000 = 0:
            plot_results(lowres_img, index, "lowres")
            plot_results(highres_img, index, "highres")
            plot_results(prediction, index, "prediction")
    
    print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / (index+1)))
    print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / (index+1)))