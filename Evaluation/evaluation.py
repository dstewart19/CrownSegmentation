# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:53:40 2020

@author: d.stewart
"""
import geopandas as gp
import numpy as np
import pandas as pd
import glob, os
from scipy.optimize import linear_sum_assignment
import rasterio

def bb_intersection_over_union(boxAA, boxBB):
    # recalculate vertices for box a and b from length weight
    boxA = boxAA.copy()
    boxB = boxBB.copy()
    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_vertex_per_plot(pl):
    site = pl.split("_")[0]
    pix_per_meter = 10
    detection_path = 'submission.csv'
    ras_path = "./NeonTreeEvaluation-master/evaluation/RGB/" + pl + ".tif"
    # read plot raster to extract detections within the plot boundaries
    raster = rasterio.open(ras_path)
    true_path = "field_crowns.csv"
    gdf = gp.read_file(detection_path)
    gdf = gdf.loc[gdf['plot_name']==pl[:-4]]
    gtf = gp.read_file(true_path, bbox=raster.bounds)
    # turn WTK into coordinates within in the image
    gdf_limits = gdf.bounds
    gtf_limits = gtf.bounds
    
    print(gdf_limits)
    print(gtf_limits)
    
    xmin = raster.bounds[0]
    ymin = raster.bounds[1]

    # length
    gdf_limits["maxy"] = (gdf_limits["maxy"] - gdf_limits["miny"]) * pix_per_meter
    gtf_limits["maxy"] = (gtf_limits["maxy"] - gtf_limits["miny"]) * pix_per_meter

    # width
    gdf_limits["maxx"] = (gdf_limits["maxx"] - gdf_limits["minx"]) * pix_per_meter
    gtf_limits["maxx"] = (gtf_limits["maxx"] - gtf_limits["minx"]) * pix_per_meter

    # translate coords to 0,0
    gdf_limits["minx"] = (gdf_limits["minx"] - xmin) * pix_per_meter
    gdf_limits["miny"] = (gdf_limits["miny"] - ymin) * pix_per_meter
    gdf_limits.columns = ["minx", "miny", "width", "length"]

    # same for groundtruth
    gtf_limits["minx"] = (gtf_limits["minx"] - xmin) * pix_per_meter
    gtf_limits["miny"] = (gtf_limits["miny"] - ymin) * pix_per_meter
    gtf_limits.columns = ["minx", "miny", "width", "length"]

    gdf_limits = np.floor(gdf_limits).astype(int)
    gtf_limits = np.floor(gtf_limits).astype(int)
    return (gdf_limits, gtf_limits, gtf.id)

field_crowns = gp.read_file('field_crowns/field_crowns.shp')
sub = gp.read_file('submission.csv')
# stems = gp.read_file('cleaned_neon_stems.csv')

list_plots = sub['plot_name']

evaluation_iou = np.array([])
itc_ids = np.array([])
# get ith plot
for pl in list_plots:
    tmpi = 0
    # get coordinates of groundtruth and predictions
    gdf_limits, gtf_limits, itc_name = get_vertex_per_plot(pl)

    # initialize IoU maxtrix GT x Detections
    iou = np.zeros((gtf_limits.shape[0], gdf_limits.shape[0]))
    for det_itc in range(gtf_limits.shape[0]):
        for obs_itc in range(gdf_limits.shape[0]):
            dets = gdf_limits.iloc[obs_itc, :].values
            trues = gtf_limits.iloc[det_itc, :].values
            # calculate the iou
            iou[det_itc, obs_itc] = bb_intersection_over_union(dets, trues)
            tmpi+=1
    # calculate the optimal matching using hungarian algorithm
    mlocs = np.argmin(-iou,axis=1)
    
    # assigned couples
    itc_ids = np.append(itc_ids, itc_name)
    foo = np.take_along_axis(iou,mlocs[:,None],axis=1)
    plot_scores = foo
    evaluation_iou = np.append(evaluation_iou, plot_scores)  # pl,plot_scores])
    print(evaluation_iou)
# concatenate the three columns and save as a csv file
