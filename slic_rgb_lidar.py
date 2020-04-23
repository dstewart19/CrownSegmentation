"Dylan Stewart 4/15/2020"

import numpy as np 
import tqdm
import scipy.io as sio
from skimage import color
from PIL import Image

def find_local_minimum(center):
    min_grad =  1
    loc_min = center
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = img[j+1, i]
            c2 = img[j, i+1]
            c3 = img[j, i]
            gradient = np.linalg.norm(c1-c3) + np.linalg.norm(c2-c3)
            if gradient < min_grad:
                min_grad = gradient
                loc_min = [i, j]

    return loc_min

def calculate_centers(par):
    centers = []
    step = par['step']
    SLIC_width = par['width']
    SLIC_height = par['height']
    for i in range(step, SLIC_width - int(step/2), step):
        for j in range(step, SLIC_height - int(step/2), step):
            nc = find_local_minimum(center=(i, j))
            colors = img[nc[1], nc[0]]
            center = [nc[0], nc[1]]
            for k in range(0, len(colors)):
                center.append(colors[k])
            centers.append(center)

    return centers

def SLIC_Multi_Parameters(K,m,Iters):
    par = {}
    par['k'] = K
    par['m'] = m
    par['iter'] = Iters
    

def SLIC_Multi(img,par):
    #get parameters for algorithm
    par['height'], par['width'] = img.shape[:2]
    SLIC_distances = 1 * np.ones(img.shape[:2])
    SLIC_clusters = -1 * SLIC_distances
    par['step'] = int((img.shape[0]*img.shape[1]/par['k'])**0.5)
#    SLIC_center_counts = np.zeros(len(calculate_centers(par['step'])))
    SLIC_centers = np.array(calculate_centers(par['step']))
    
    #run main loop
    indnp = np.mgrid[0:par['height'],0:par['width']].swapaxes(0,2).swapaxes(0,1)
    for i in tqdm.tqdm(range(par['iter'])):
        SLIC_distances = np.full(img.shape[:2], np.inf)
        for j in range(SLIC_centers.shape[0]):
            x_low, x_high = int(SLIC_centers[j][0] - par['step']), int(SLIC_centers[j][0] + par['step'])
            y_low, y_high = int(SLIC_centers[j][1] - par['step']), int(SLIC_centers[j][1] + par['step'])

            if x_low <= 0:
                x_low = 0

            if x_high > par['width']:
                x_high = par['width']

            if y_low <=0:
                y_low = 0

            if y_high > par['height']:
                y_high = par['height']

            cropimg = img[y_low : y_high , x_low : x_high]
            color_diff = cropimg - img[int(SLIC_centers[j][1]), int(SLIC_centers[j][0])]
            colordiff_sq = np.square(color_diff)
            colordiff_sum = np.sum(colordiff_sq, axis = 3)
            color_dist = np.sqrt(colordiff_sum)
            color_distSum = np.sum(color_dist, axis = 2)
            
            yy, xx = np.ogrid[y_low : y_high, x_low : x_high]
            pixdist = ((yy-SLIC_centers[j][1])**2 + (xx-SLIC_centers[j][0])**2)**0.5

            # SLIC_m is "m" in the paper, (m/S)*dxy
            dist = color_distSum + ((par['m']/par['step'])*pixdist)

            distance_crop = SLIC_distances[y_low : y_high, x_low : x_high]
            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]
            SLIC_distances[y_low : y_high, x_low : x_high] = distance_crop
            SLIC_clusters[y_low : y_high, x_low : x_high][idx] = j

        for k in range(len(SLIC_centers)):
            idx = (SLIC_clusters == k)
            featurenp = img[idx]
            distnp = indnp[idx]
            ind = 0
            for n in range(2,len(SLIC_centers[k])):
                SLIC_centers[k][n] = np.sum(featurenp, axis=0)[ind]
                ind += 1
            sumy, sumx = np.sum(distnp, axis=0)
            SLIC_centers[k][0:2] = sumx, sumy
            SLIC_centers[k] /= np.sum(idx)
    return par, SLIC_clusters, SLIC_distances

# global variables
im1 = r"C:\Users\d.stewart\DEEPFOREST\images\2\1.tif"
im2 = r"C:\Users\d.stewart\DEEPFOREST\images\2\2.tif"

Im1 = np.array(Image.open(im1))
Im2 = np.array(Image.open(im2))

lab1 = color.rgb2lab(Im1)
lab2 = color.rgb2lab(Im2)
lab1 = np.expand_dims(lab1,axis=2)
lab2 = np.expand_dims(lab2,axis=2)
img = np.concatenate((lab1,lab2),axis=2)

# main
par = SLIC_Multi_Parameters(500,1,100)
pars,SLIC_clusters, SLIC_distances = SLIC_Multi(img,par)
sio.savemat(r"C:\Users\d.stewart\DEEPFOREST\Superpixels\2" + "K="+str(par['k']) +"m="+ str(par['m'])+"I="+str(par['iter']) + ".mat", {'labels':SLIC_clusters})