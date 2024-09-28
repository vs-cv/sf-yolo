import numpy as np
import torch
from TargetAugment.enhance_vgg16 import enhance_vgg16

def get_style_images(im_data, args, adain=None):
    if adain==None:
        adain = enhance_vgg16(args)
    # Apply style to the image, use save_images=True to save the images (useful for debugging)
    styled_im_data = im_data * 0 + 1 * adain.add_style(im_data, 0, save_images=args.save_style_samples)
    
    return styled_im_data
