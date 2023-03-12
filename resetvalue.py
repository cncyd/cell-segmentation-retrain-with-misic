#Modify the result directly identified from the fluorescent image as the label for the identification of the bright field image

import os
#from numpy import array
import numpy as np
'''
from pylab import *
from skimage.exposure import adjust_gamma
from skimage.util import view_as_windows
from skimage.segmentation import watershed
from scipy.ndimage import label
from skimage.transform import resize,rescale
from skimage.io import imread,imsave
from skimage.feature import shape_index
import random
'''
def stitch_tiles(patches,params):
    ''' stitch tiles generated from extract tiles '''
    size = params['size']
    pad_row = params['pad_row']
    pad_col = params['pad_col']
    im_size = params['im_size']
    positions = params['positions']
    exclude = params['exclude']
       
   
    result = np.zeros((im_size[0],im_size[1],patches.shape[-1]))*1.0
    #print(result)
   
    for i,pos in enumerate(positions):        
        rr,cc = pos[0],pos[1]    
        result[rr:rr+size,cc:cc+size,:] = patches[i,exclude:-exclude,exclude:-exclude,:]*1.0
        #print(result.round(1))
    #print('finished！')
    if pad_row>0:
        result = result[:-pad_row,:]
    if pad_col>0:
        result = result[:,:-pad_col]
    #print(result)    
    return result

filepath='F:\\imageprocess\\'
namelist=os.listdir(filepath + 'focuslabel')
fullenth=len(namelist)
count=0
while count<fullenth:
    #binarize the recognition result as a label
    pic=namelist[count]
    patches=np.load('F:/imageprocess/focuslabel/'+pic)
    patches[:,:,:,0][ patches[:,:,:,0]<0.4]=0
    patches[:,:,:,0][ patches[:,:,:,0]>=0.4]=1
    patches[:,:,:,1][ patches[:,:,:,1]>0.1]=1
    patches[:,:,:,1][ patches[:,:,:,1]<=0.1]=0
    # define a place to save the label
    tar='/'+pic
    tarpath=os.path.join(filepath,'newfocuslabel'+tar)
    patches=patches.astype(int)
    #Use statistics to monitor each label's situation
    list=patches[:,:,:,0].tolist()
    intervals = {'{}-{}'.format(x/4,(x+1)/4):0 for x in range(4)}
    for m in list:
        for n in m:
            for b in n:
                for interval in intervals:
                    start,end = tuple(interval.split('-'))
                    if float(start)<=b<=float(end):
                        intervals[interval] +=1
    #print each label's monitor result
    print('------------------'+str(pic)+'---------------------')
    print(intervals)
    np.save(tarpath,patches)
    print('saved')
    count+=1

