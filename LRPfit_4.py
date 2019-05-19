#! /usr/bin/env python

from time import time #, strftime, gmtime
import numpy as np
from paraboloid import paraboloid as pb
#from numbapro import vectorize
from numba import vectorize
import math
from frange import frange
from tkinter import Tk
from tkinter import filedialog

#import matplotlib.pyplot as plt5
#from frange import frange as fr

###################### PARAMETERS ##################################
#datafile='averaged_flat_ftrd_flat_cln_cln.txt' # file containing the data
#rng=frange(2,4.1,0.5) # mask diameter range
temp = input('Enter a range (e.g. 0.5,0.5,10) default=[quit]: ')
if temp=='':
    raise ValueError('Cancelled...')
try:
    temp=temp.split(',')
    rng=frange(float(temp[0]),float(temp[2])+float(temp[1]),float(temp[1])) # mask diameter range
except:
    print('Please suply three values separated by a comma. Ex:')
    temp = input('Please suply three values separated by a comma: min,step,max. Ex: 0.5,0.5,10 ')
    if temp=='':
        raise ValueError('Cancelled...')
    temp=temp.split(',')
    rng=frange(float(temp[0]),float(temp[2])+float(temp[1]),float(temp[1])) # mask diameter range

root = Tk()
root.withdraw()
root.update() #prevent bug
datafile=filedialog.askopenfilename(filetypes=[("Text files","*.txt")],title = "Select data file")
if datafile=='':
    raise ValueError('Cancelled...')
else:
    print(datafile)

#meshfile='mesh.txt' # file containing the top node coordinates (can be the same as above, if unavailable)
meshfile=filedialog.askopenfilename(filetypes=[("Text files","*.txt")],title = "Select mesh file")
if meshfile=='':
    raise ValueError('Cancelled...')
else:
    print(meshfile)
root.update()#prevent bug
root.destroy()

####################################################################

@vectorize(['float32(float32, float32, float32, float32, float32, float32)',
            'float64(float64, float64, float64, float64, float64, float64)'],
           target='parallel')
def mm(x,y,x_grid,y_grid,a,b):
    return math.sqrt(((x-x_grid)/(a/2))**2+((y-y_grid)/(b/2))**2)

def lrp(x,y,z,mask_size,*args):
    #[Data_out] = splfit_2(x_data,y_data,z_data,mask_size,varargin)
    
    mask_size=np.asarray(mask_size)
    
    if mask_size.size==1:
        a = mask_size #Horizontal size of the mask
        b = mask_size #Vertical size of the mask
    elif mask_size.size==2:
        a = mask_size[0]
        b = mask_size[1]
    else:
        print('Bad mask size.\n    Please inform a real number or a matrix in the form: [x y]')
    
    if len(args)==0:
        x_grid=x
        y_grid=y
        print('Fitting all points.')
    else:
        x_grid=args[0]
        y_grid=args[1]
        print('Using provided grid.')
    
    #if ((a<3) or (b<3)): #Mask too small or invalid.
    #    print('Please inform a mask matrix of at least 3x3.')
    #elif (a>=np.max(x)-np.min(x)) or (b>=np.max(y)-np.min(y)):
    if False: #(a>=np.max(x)-np.min(x)) or (b>=np.max(y)-np.min(y)):
        print('Mask dimenions are too large. Consider reducing.')
    else:
        
        mask_matrix = np.asarray([False for i in x])
        mask=np.zeros_like(x)
        Data_out=np.asarray([float('nan') for i in x_grid])
        perc_done=0
        prev_perc_done=0
        starttime=time()
        zNonNaNs=np.invert(np.isnan(z))
        for k in range(0,x_grid.size):
            k=int(k)
            
            #Time estimate (best before so that it will be always shown)
            perc_done=perc_done=np.floor(k/x_grid.size*100)/1
            if perc_done!=prev_perc_done:
                print(str(perc_done) + '% completed - Estimated time to finish: ' + str((time()-starttime)/perc_done*(100-perc_done)/60) + ' min')
                prev_perc_done=perc_done
            
            #Calculating mask and skipping if nearest valid point is too far (arbitrary)
            mask=mm(x,y,x_grid[k],y_grid[k],a,b)
            mask_matrix=(mask<=1)*zNonNaNs #making mask round
            
#            if np.min(mask[zNonNaNs])<=0.5/math.sqrt(a**2 + b**2): # ~0.5 pspacing away from the nearest valid point
#                mask_matrix=(mask<=1)*zNonNaNs #making mask round
#            else:
#                continue
                     
            #skipping if only a few points within the mask are not nans.
            fact=1
            while z[mask_matrix].shape[0]<=100:
                print('Increasing mask for point ', str(k))
                fact=fact*math.sqrt(2)
                mask_matrix=(mask<=1*fact)*zNonNaNs #making mask round
    
            # Fitting paraboloid
            try:
                #t2 = time()
                Data_out[k]=pb(x[mask_matrix],y[mask_matrix],z[mask_matrix],x_grid[k],y_grid[k])
                #aa=x[mask_matrix]
                #bb=y[mask_matrix]
                #cc=z[mask_matrix]
                #Data_out[k]=float('nan')
                #print('LRPfit: ' + str((time() - t2)*1000) + ' ms')
            except:
                Data_out[k]=float('nan')
            #Data_out[k]=z[k]
            #disp(['Loop count: ' num2str(k1)])

            
            #if k>=10: #just to check the code below!
            #    break
        if Data_out[np.isnan(Data_out)].shape[0]/x_grid.shape[0]>0.1:
            print('Not enough valid points, at current state...')
            return Data_out
                
        #Now filling the NaNs in Data_out
        x=x_grid
        y=y_grid
        keepLooping=Data_out[np.isnan(Data_out)].shape[0]
        if keepLooping > 0: print(keepLooping, 'NaNs in the data. Inter/extrapolating the data to fill gaps.')
        a=a*math.sqrt(2)
        b=b*math.sqrt(2)
        mask_matrix = np.asarray([False for i in x_grid])
        while keepLooping:
            z=Data_out
            zNonNaNs=np.invert(np.isnan(z))
            for k in range(0,x_grid.size):
                if not np.isnan(z[k]): #Skipping if point has already been smoothed
                    continue
                k=int(k)
                
                #Calculating mask
                mask_matrix=(mm(x,y,x_grid[k],y_grid[k],a,b)<=1)*zNonNaNs
                
                #skipping if only a few points within the mask are not nans.
                if z[mask_matrix].shape[0]<=15:
                    print(z[mask_matrix].shape[0],'- Skipping...')
                    continue
                
                # Fitting paraboloid
                try:
                    Data_out[k]=pb(x[mask_matrix],y[mask_matrix],z[mask_matrix],x_grid[k],y_grid[k])
                except:
                    Data_out[k]=float('nan')
            
            if keepLooping==Data_out[np.isnan(Data_out)].shape[0]:
                a=a*math.sqrt(2) #hard-coded increment. Increases mask when not converging.
                b=b*math.sqrt(2) #hard-coded increment. Could be relative if pt spacing was available
                print(keepLooping,'NaNs, increasing mask to:',a,'x',b,'. This may take a while... Please wait...')
            else:
                keepLooping=Data_out[np.isnan(Data_out)].shape[0]
                print(keepLooping, 'NaNs, making progress... Please wait...')
                
            
    return Data_out

#f=open('QuenchedCylinder_SurfaceDeformation_total_finer_noise.txt','r')
f=open(datafile,'r')
data=f.read()
f.close()
data=data.replace('\n\n','\n')
data=data.split()
#converting str to floats and splitting in x, y and z
x=np.asarray([float(i) for i in data[0::3]])
y=np.asarray([float(i) for i in data[1::3]])
z=np.asarray([float(i) for i in data[2::3]])

#load grid from coarser data
f=open(meshfile,'r')
data=f.read()
f.close()
data=data.replace('\n\n','\n')
data=data.split()
#converting str to floats and splitting in x, y and z

#rng=np.array([float(i) for i in rng])
if 1: # if 0: # 
    for mask_size in rng:
        x_grid=np.asarray([float(i) for i in data[0::3]])
        y_grid=np.asarray([float(i) for i in data[1::3]])
        #z_grid=np.asarray([float(i) for i in data[2::3]])
        #x_grid=x_grid[np.invert(np.isnan(z_grid))]
        #y_grid=y_grid[np.invert(np.isnan(z_grid))]
        #t = time()
        #mask_size=np.array([ mask_size/1.4142, mask_size*1.4142 ]) # elliptic mask
        #mask_size=np.array([ mask_size, mask_size*1.4]) # elliptic mask
        zNonNaNs=np.invert(np.isnan(z))
        Data_out=lrp(x[zNonNaNs],y[zNonNaNs],z[zNonNaNs],mask_size,x_grid,y_grid)
        #print(time() - t)
        d_out=np.vstack((x_grid, y_grid, Data_out)).T
        print('Saving ' + datafile.rsplit('.',1)[0] + "_LRP-" + str(mask_size) + '.txt ...')
        np.savetxt(datafile.rsplit('.',1)[0] + "_LRP-" + str(mask_size) + '.txt', d_out, delimiter=" ")
        #fi=plt.tricontourf(x_grid,y_grid,Data_out)
        #plt.savefig('LRP_' + str(mask_size) + '_Rail_noise_cln.png')
else:
    x_grid=np.asarray([float(i) for i in data[0::3]])
    y_grid=np.asarray([float(i) for i in data[1::3]])
    mask_size=1.8/1.0
    
    zNonNaNs=np.invert(np.isnan(z))
    Data_out=lrp(x[zNonNaNs],y[zNonNaNs],z[zNonNaNs],mask_size,x_grid,y_grid)
    #print(time() - t)
    d_out=np.vstack((x_grid, y_grid, Data_out)).T
    np.savetxt("LRP_" + str(mask_size) + datafile, d_out, delimiter=" ")
    #fi=plt.tricontourf(x_grid,y_grid,Dat
    
    
    #t = time()
    #Data_out=lrp(x,y,z,mask_size,x_grid,y_grid)
    #print(time() - t)
    #fi=plt.tricontourf(x_grid,y_grid,Data_out)