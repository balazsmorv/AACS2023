#!/usr/bin/env python3

import os
import PIL
from PIL import Image
from tqdm import tqdm

input_path = '/Users/balazsmorvay/Downloads/val_images'
output_path = '/Users/balazsmorvay/Downloads/val_images_lowres'

upscale_factor = 4

os.mkdir(output_path)

for filename in tqdm(os.listdir(input_path)):
	if filename == '.DS_Store':
		continue
	f = os.path.join(input_path, filename)
	if os.path.isfile(f):
		img = Image.open(f)
		lowres_img = img.resize((img.size[0] // upscale_factor, img.size[1] // upscale_factor), PIL.Image.BICUBIC)
		lowres_img.save(os.path.join(output_path, filename))
		#print(f"Saved {f}")
	else:
		print(f"Error with file {f}")
		