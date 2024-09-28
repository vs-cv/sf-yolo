from shutil import copy
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--images_folder', type=str, required=True,
                    help='The path to the images folder')
parser.add_argument('--scenario_name', type=str, required=True,choices=['city2foggy'],help='The name of the scenario')
parser.add_argument('--image_suffix', type=str, default='jpg',help='image suffix')

args = parser.parse_args()

images_path=args.images_folder
dataset=args.scenario_name

dir_images=os.path.join('data',dataset)

dir=os.path.join('data',dataset)
if not os.path.exists(dir):
    os.makedirs(dir)

subfolders = ['train']

for subfolder in subfolders:
    subfolder_images_path = os.path.join(images_path, subfolder)
    image_files = os.listdir(subfolder_images_path)

    for image_file in tqdm(image_files):
        if image_file.endswith(args.image_suffix):
            base_filename = os.path.splitext(image_file)[0]
            copy(os.path.join(subfolder_images_path, image_file), os.path.join(dir_images, image_file))