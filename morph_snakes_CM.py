# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:40:00 2020

@author: d.stewart
"""

#Clear workspace
import os
clear = lambda: os.system('cls')
clear()
"""
%=====================================================================
%======================= Import Packages =============================
%=====================================================================
"""
import glob
import re
import struct
import math
import json
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import animation
import matplotlib.patches as patches
import scipy as sp
from sklearn.feature_extraction import image as skIm
from skimage import img_as_float
from skimage import measure
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set,
                                  circle_level_set)
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage import io
import argparse
from skimage import exposure
from sklearn import preprocessing
from astropy.stats import median_absolute_deviation

def morphSnakes(fp, frame, alph, sig, thresh):
    """{ 
    ******************************************************************
    *  Fnc:     read_arf() 
    *  Desc:    reads an arf file and returns dictionary of parsed info
    *  Inputs:  fp - full path to arf file
    *  Outputs: 
    ******************************************************************
    }"""
    ############################ Read Public Release Data ##################################################
#    print('Reading data...')
    _, fn = os.path.split(fp)
    basename, _ext = os.path.splitext(fn)
    df = pd.read_csv(f"{DATA_DIR}/Metric/{basename}.bbox_met", header=None, names=[
        "site", "unknown1", "unknown2", "sensor", "scenario", "frame", "ply_id", "unknown3", "unknown4", "upperx",
        "uppery", "unknown5", "unknown6", "unknown7", "unknown8", "unknown9", "unknown10",
        "unknown11"
    ])
    agt = read_agt(f"{DATA_DIR}/cegr/agt/{basename}.agt")
    f = open(fp, "rb")
    header = f.read(8 * 4)
    header = list(struct.iter_unpack(">I", header))
    fptr = np.memmap(fp, dtype="uint16", mode='r', shape=(header[5][0], header[2][0], header[3][0]), offset=32)
    frames, cols, rows = fptr.shape
#    print('Data loaded!')
    im = fptr[frame].T.byteswap()
    tgtx, tgty = map(int, agt['Agt']['TgtSect'][f'TgtUpd.{frame}'][f'Tgt.{frame}']['PixLoc'].split())
    upper_left_x, upper_left_y = df[df['frame'] == frame + 1][['upperx', 'uppery']].iloc[0]
    tgt_width = 2 * (tgtx - upper_left_x)
    tgt_height = 2 * (tgty - upper_left_y)
    ######################################### Find BB with GAC ################################################
    def store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """
        def _store(x):
            lst.append(np.copy(x))
        return _store
    plt.ioff()
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()
    # Morphological GAC
    image = img_as_float(im.T)
    gimage = inverse_gaussian_gradient(image, alpha=alph, sigma=sig)
    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("MWIR", fontsize=12)
    ax[1].imshow(gimage, cmap="gray")
    ax[1].set_axis_off()
    ax[1].set_title("Inverse Gaussian Gradient", fontsize=12)
    ######################################## Set Initial Contour ###########################################
    ## Here you will want to set it as the bounding box, then you can set the morph snake to shrink instead
    ## of dialate.
    ########################################################################################################
    # Initial level set
    init_ls = circle_level_set(image.shape, center=(tgty,tgtx), radius=5)
#    init_ls[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ## Initialize the Morph Snake,  you will want it to shrink (balloon=-1)
    ls = morphological_geodesic_active_contour(gimage, 230, init_ls,
                                               smoothing=1, balloon=1,
                                               threshold=thresh,
                                               iter_callback=callback)
    ax[2].imshow(image, cmap="gray")
    ax[2].set_axis_off()
    ax[2].contour(ls, [0.5], colors='r')
    ax[2].set_title("Center Point GAC Segmentation", fontsize=12)
    ax[3].imshow(ls, cmap="gray")
    ax[3].set_axis_off()
    contour = ax[3].contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax[3].contour(evolution[100], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 100")
    contour = ax[3].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 230")
    ax[3].legend(loc="upper right")
    title = "Center Point GAC Evolution"
    ax[3].set_title(title, fontsize=12)

    fig.tight_layout()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Morph Snakes, alpha={alph}, sigma={sig}, threshold={thresh}')
#    plt.savefig(f"{DATA_DIR}\\morphSnakes\\{basename}_{alph}_{sig}_{thresh}.png")
#    plt.close(fig)
    plt.show()
def run_morph_snakes():
    """{ 
    ******************************************************************
    *  Fnc:     read_arf() 
    *  Desc:    reads an arf file and returns dictionary of parsed info
    *  Inputs:  fp - full path to arf file
    *  Outputs: 
    ******************************************************************
    }"""
    ########### Load a single arf file ######
    frame = 30
    for fp in glob.glob(f"{DATA_DIR}\\cegr\\arf\\*.arf"):
        if (0 <= (fp.find("cegr02003_0009"))):
            print(fp)
            for a in [200]:
                for s in [1.5]:
                    for t in [0.95]:
                        try:
                            morphSnakes(fp,frame, a, s, t)
#                            print(a)
                        except:
                            pass