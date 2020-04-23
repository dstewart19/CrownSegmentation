# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:41:23 2020

@author: d.stewart
"""
import rasterio
import numpy as np
from affine import Affine
from pyproj import Proj, transform

RGB = 'TEAK_2018_RGB/TEAK_053.tif'
LIDAR = 'TEAK_2018_CHM/TEAK_053_CHM.tif'

#get eastings and northings of RGB
with rasterio.open(RGB) as r:
    T0 = r.transform
    p1 = Proj(r.crs)
    A = r.read()

cols, rows = np.meshgrid(np.arange(A.shape[2]),np.arange(A.shape[1]))
T1 = T0*Affine.translation(0.5,0.5)
rc2en = lambda r, c: (c,r) * T1

eastings, northings = np.vectorize(rc2en,otypes=[np.float,np.float])(rows,cols)

#RGB cube
A = np.moveaxis(A,0,-1)
A = np.reshape(A,((A.shape[0]*A.shape[1],A.shape[2])))

#get eastings and northings of LIDAR
with rasterio.open(LIDAR) as r:
    T0_L = r.transform
    p1_L = Proj(r.crs)
    A_L = r.read()

cols_L, rows_L = np.meshgrid(np.arange(A_L.shape[2]),np.arange(A_L.shape[1]))
T1_L = T0_L*Affine.translation(0.5,0.5)
rc2en_L = lambda r, c: (c,r) * T1_L

eastings_L, northings_L = np.vectorize(rc2en_L,otypes=[np.float,np.float])(rows_L,cols_L)

#LIDAR cube
A_L = np.moveaxis(A_L,0,-1)
A_L = np.reshape(A_L,((A_L.shape[0]*A_L.shape[1],A_L.shape[2])))