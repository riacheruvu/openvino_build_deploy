import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import openvino as ov

#LCM image output 
h = 512  
w = 512 


def superres_load(model_path, device):
    core = ov.Core()
    model = core.read_model(model=model_path)
    original_image_key, bicubic_image_key = model.inputs

    input_height, inputwidth = list(original_image_key.shape)[2:]
    target_height, target_width = list(bicubic_image_key.shape)[2:]
    upsample_factor = int(target_height / input_height)


    shapes = {}
    for input_layer in model.inputs:
        if input_layer.names.pop() == "0":
            shapes[input_layer] = input_layer.partial_shape

            shapes[input_layer][2] = h
            shapes[input_layer][3] = w
        elif input_layer.names.pop() == "1":
            shapes[input_layer] = input_layer.partial_shape
            shapes[input_layer][2] = upsample_factor * h
            shapes[input_layer][3] = upsample_factor * w

    model.reshape(shapes)


    compiled_model = core.compile_model(model=model, device_name=device)
    return compiled_model,upsample_factor