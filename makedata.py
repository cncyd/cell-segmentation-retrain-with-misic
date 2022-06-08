import numpy as np
from pylab import *
from skimage.exposure import adjust_gamma
from skimage.util import view_as_windows
from skimage.segmentation import watershed
from scipy.ndimage import label
#from tensorflow.keras.models import load_model
import os
import sys
import numpy as np
from skimage.transform import resize,rescale
from skimage.io import imread,imsave
from skimage.feature import shape_index


def shapeindex_preprocess(im):
    ''' apply shap index map at three scales'''
    sh = np.zeros((im.shape[0],im.shape[1],3))
    if np.max(im) ==0:
        return sh
    
    # pad to minimize edge artifacts                    
    sh[:,:,0] = shape_index(im,1, mode='reflect')
    sh[:,:,1] = shape_index(im,1.5, mode='reflect')
    sh[:,:,2] = shape_index(im,2, mode='reflect')
    #sh = 0.5*(sh+1.0)
    return sh

def normalize2max(im):
    ''' normalize to max '''
    im = im-np.min(im)
    return im/np.max(im)

def extract_tiles(im,size = 512,exclude = 12):
    ''' extract tiles from image of size 'size' to be stiched back such that 'exclude' pixels near border of tile are excluded '''
    size = size-2*exclude

    if len(im.shape)<3:
        im = im[:,:,np.newaxis]
    sr,sc,ch = im.shape 
    
    pad_row = 0 if sr%size == 0 else (int(sr/size)+1) * size - sr
    pad_col = 0 if sc%size == 0 else (int(sc/size)+1) * size - sc
    im1 = np.pad(im,((0,pad_row),(0,pad_col),(0,0)),mode = 'reflect')
    sr1,sc2,_ = im1.shape
    

    rv = np.arange(0,im1.shape[0],size)
    cv = np.arange(0,im1.shape[1],size)
    cc,rr = np.meshgrid(cv,rv)
    positions = np.concatenate((rr.ravel()[:,np.newaxis],cc.ravel()[:,np.newaxis]),axis = 1)
        
    im1 = np.pad(im1,((exclude,exclude),(exclude,exclude),(0,0)),mode = 'reflect')

    params = {}
    params['size'] = size
    params['exclude'] = exclude
    params['pad_row'] = pad_row
    params['pad_col'] = pad_col
    params['im_size'] = [sr1,sc2]
    params['positions'] = positions
    
    patches = view_as_windows(im1,(size+2*exclude,size+2*exclude,ch),size)    
    patches = patches[:,:,0,:,:,:]
    patches = np.reshape(patches,(-1,patches.shape[2],patches.shape[3],patches.shape[4]))
    return patches,params

def stitch_tiles(patches,params):
    ''' stitch tiles generated from extract tiles '''
    size = params['size']
    pad_row = params['pad_row']
    pad_col = params['pad_col']
    im_size = params['im_size']
    positions = params['positions']
    exclude = params['exclude']
        
    
    result = np.zeros((im_size[0],im_size[1],patches.shape[-1]))*1.0
    
    
    for i,pos in enumerate(positions):        
        rr,cc = pos[0],pos[1]    
        result[rr:rr+size,cc:cc+size,:] = patches[i,exclude:-exclude,exclude:-exclude,:]*1.0
    
    if pad_row>0:
        result = result[:-pad_row,:]
    if pad_col>0:
        result = result[:,:-pad_col]
    return result

def postprocess_ws(im,yp):
    '''Watershed based postprocessing using image and its pixel probabilities'''
    # mask dilated
    mask = (yp[:,:,0] > 0.4)
    # watershed potential
    d = shape_index(im,1.5,mode = 'reflect')        
    # markers
    # get poles from contour predictions as markers
    sh = shape_index(yp[:,:,1],1,mode = 'reflect')
    markers,c = label(yp[:,:,0] > 0.95)
    # ther markers should be unique to each cell 
    markers = markers*(sh<-0.5)  # only poles    
    ws = watershed(d, markers=markers,watershed_line = True,mask = mask,compactness = 1,connectivity = 1)        
    return ws   

#nowpath=os.getcwd()


py_path=sys.argv[0]
#nowpath=os.path.dirname(py_path)
#path=os.path.join(nowpath,'image','train','image')
#print(path)
nowpath=r'F:\\imageprocess'
path=r'F:\\imageprocess\\image\\train\\image'
files_list = os.listdir(path)
triger=False
for file in files_list:
    filename, suffix = os.path.splitext(file)
    if filename!='51':
        continue
    file_path = path + "/" + filename + '.tif'
    im = imread(file_path)
    sr,sc = im.shape
    im=adjust_gamma(im,0.41)
    mean_width = 7
    scale = (10/mean_width)
    img = rescale(im,scale,preserve_range = True)
    img = normalize2max(img)
    img=shapeindex_preprocess(img)
    tiles,params = extract_tiles(img,size = 256,exclude = 16) 
    # model_name = os.path.join(os.path.dirname(__file__), 'MiSiDC04082020.h5')
    # model = load_model(model_name,compile=False)
    # model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # yp = model.predict(tiles)
    tar='/'+filename+'.npy'
    #lab='label/'+filename+'.npy'
    tarpath=os.path.join(nowpath,'target'+tar)
    #print(tarpath)
    np.save(tarpath,tiles)
    #np.save(os.path.join(nowpath,lab),yp)
    print(filename)