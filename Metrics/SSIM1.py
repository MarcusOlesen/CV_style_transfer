import os
import re
import numpy as np
from PIL import Image
import cv2
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import shutil

def compute_ssim(path1, path2):
    # load as H×W×3 uint8 arrays
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    dim = (512, 512)
    img1 = cv2.resize(img1, dim)
    img2 = cv2.resize(img2, dim)

    # multichannel=True lets SSIM run on color
    score = ssim(img1, img2, channel_axis=2, data_range=img1.max()-img1.min())
    return score

def stats_ssim_from_folder(folder, outname, model="AdaAttN"):
    content_dir = "../data/content_test_small"
    #content_dir = "../StyTR-2/input/content"
    pattern = re.compile(r"(.+)_cs\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)
    out = []
    
    for fname in sorted(os.listdir(folder)):
        if model == "AdaAttN":
            m = pattern.match(fname)
            if not m:
                continue
                
            stylized_path = os.path.join(folder, fname)
            content_path = stylized_path.replace("_cs.", "_c.")
            
        elif model=="StyTr2":
            if not fname.lower().endswith(".jpg"):
                continue  # Skip non-JPG files
                
            stylized_path = os.path.join(folder, fname)
            content_path = os.path.join(content_dir, 
                                        re.match(r'^(.*?)_stylized_', fname).group(1) + ".jpg")
            

        if not os.path.isfile(content_path):
            print(f"  → skipping {fname}: missing content file")
            continue

        try:
            score = compute_ssim(stylized_path, content_path)
        except Exception as e:
            print(f"  ⚠️ failed on {fname}: {e}")
            continue

        out.append(score)
    
    with open("SSIM_" + outname + ".txt", "w") as f:
        f.write(str(np.mean(out)) + "\n")

stats_ssim_from_folder("../StyTR-2/out_selftrained", "StyTr2_retrain", model="StyTr2")