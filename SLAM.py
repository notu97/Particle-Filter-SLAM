#!/usr/bin/env python
# coding: utf-8

# In[4]:


import load_data 
import numpy as np
import cv2 
from scipy import io
import matplotlib as plt
import matplotlib.pyplot as pl
import time
# odom=load_data.get
import Transform as Tfm
from mpl_toolkits.mplot3d import Axes3D
from p2_utils import *
# import bresenham as bres
from IPython.display import clear_output
import cv2
from skimage.draw import line
import scipy


# In[23]:


# Load data here
laser= load_data.get_lidar('Data/lidar/train_lidar4')
jointAngle=load_data.get_joint('Data/joint/train_joint4')


# In[5]:


# Joint angle and Odometry Time synchronization. 

# Get all timestamps from Laser data Dictionary and store in 'a'
a=[]
for i in range(len(laser)):
    a.append(laser[i]['t'][0][0])
    
# Find the time stamp nearest to all the times in array 'a' from the jointAngle data and store in array 'b'
b=[]
# ind=0
for i in a:
    b.append(np.argmin( abs(jointAngle['ts'][0]-i)))   


# In[6]:


# This code was used for downsampling, not needed when going step by step
pos=[]
for i in range(len(laser)):
    pos.append((laser[i]['delta_pose'][0]).T)
position=np.cumsum(pos,axis=0).T   


# In[7]:


# Lidar to body frame tranformation
def lidar2body(scan,neck_angle,head_angle,indValid):
    '''
     To tranform LIDAR coordinate to Body frame coordinate\\
     Scan: Array of Lidar Scan 1 x 1081 (in meters)\\
     head_angle: Scalar (in radians)\\
     neck_angle: Scalar (in radians)
    '''
    d=np.zeros((4,len(scan.T)))
    angles=np.deg2rad(np.arange(-135,135.25,0.25))
    angles=angles[indValid]
    # print('Scan size: ',(scan.T).shape)
    # print('angles :' ,(angles.T).shape)

    cos=np.multiply(scan,np.cos(angles))
    sin=np.multiply(scan,np.sin(angles))

    cos=cos.reshape((1,np.sum(indValid)))
    sin=sin.reshape((1,np.sum(indValid)))

    # print(np.shape(cos),np.shape(sin))

    z=np.zeros( (1,len(scan.T)) )
    on=np.ones( (1,len(scan.T)) )
    add=np.concatenate((z,on),axis=0)
    # print(add.shape)
    add2=np.concatenate((cos,sin),axis=0)
    # print(np.shape(add), np.shape(add2))
    
    
    lid_coord=np.concatenate((add2,add),axis=0)
    # print(lid_coord.shape)
    for i in range(0,len(lid_coord.T)):
        p=np.array([[0],[0],[0.48]])
        a=np.array([[np.cos(neck_angle),-np.sin(neck_angle),0],
                   [np.sin(neck_angle), np.cos(neck_angle),0],
                   [0,0,1]])

        b=np.array([[np.cos(head_angle),0,np.sin(head_angle)],
                   [0,1,0],
                   [-np.sin(head_angle),0,np.cos(head_angle)]])
        c=np.matmul(b,a) 
        c=np.concatenate((c,p),axis=1) 
        c=np.concatenate((c,np.array([[0,0,0,1]])),axis=0)
        d[:,i]=np.matmul(c,lid_coord[:,i])        
    return d


# In[8]:


# Map correlation function
def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)),                                                             np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
            scan_map = np.zeros_like(im)
            scan_map[ix[valid], iy[valid]] = True
            cpr[jx, jy] = np.sum(np.logical_and(im, scan_map))
    return cpr


# In[18]:


# Define MAP and other variables
N, N_threshold =100,35 # 30 # 35 # No of Particles
X=np.zeros((N,3))

# X[:,2]=X[:,2]/10
W=(1.0/N)*np.ones(N)
var=0.05

MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -20  #meters
MAP['ymin']  = -20
MAP['xmax']  =  20
MAP['ymax']  =  20 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
MAP['logOdds']=0*np.ones((MAP['sizex'],MAP['sizey']))
MAP['display']=0*np.ones((MAP['sizex'],MAP['sizey']),dtype=np.int8)



coordX=[]
coordY=[]
world_coords=np.zeros((4,1081))
# robo_pos=np.array([0,0,0])

x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

x_range = np.arange(-0.05, 0.06, 0.05)
y_range = np.arange(-0.05, 0.06, 0.05)


# In[ ]:


# Start For Loop to perform SLAM

get_ipython().run_line_magic('matplotlib', 'inline')
for i in np.arange(0,len(laser),1):
    print(i)
    lid=laser[i]['scan']
    pose=laser[i]['delta_pose']
    indValid = np.logical_and(( lid[0]< 30),(lid[0]> 0.1))
    lid = (lid[0][indValid])

    lid2bod=lidar2body(lid,jointAngle['head_angles'][0,b[i]],jointAngle['head_angles'][1,b[i]], indValid)

    noise =  np.tile(np.random.normal(0, 0.05, (N, 1)),(1,3))
    noise[:,2] = noise[:,2]*2
    X =  (pose[0]+X+noise).astype(float) 
    
    corr = np.zeros(N)
    corr_max=np.zeros(N)
    
    for j in range(N):
        World_coord_part=Tfm.body2world(X[j,:])@lid2bod
        World_coord_part=World_coord_part[:,World_coord_part[2,:]>0.2]
        c = mapCorrelation(MAP['map'],x_im,y_im, World_coord_part[0:2,:],x_range,y_range)
        ind = np.argmax(c)
        corr[j] = np.max(c)
        X[j, 0] += x_range[int(ind/3)]
        X[j, 1] += y_range[int(ind%3)]
    
    corr_max=(np.multiply(W, np.exp((np.array(corr)-max(corr))))).astype(float)
    W = (np.divide(corr_max, sum(corr_max))).astype(float)
    
    best_particle = np.argmax(W) # Get index of best particle
    best_world_coords= Tfm.body2world(X[best_particle,:])@lid2bod
    
    rob_grid_poseX=np.ceil((X[best_particle,0] - 1*MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    rob_grid_poseY=np.ceil((X[best_particle,1] - 1*MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    coordX.append(rob_grid_poseX)
    coordY.append(rob_grid_poseY)
    
    xis = np.ceil((best_world_coords[0,np.where( (best_world_coords[2,:])>0.2)] - 1*MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((best_world_coords[1,np.where( (best_world_coords[2,:])>0.2)] - 1*MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
     # take valid world-coords
    indGood = np.logical_and(np.logical_and(np.logical_and((xis[0] > 1), (yis[0] > 1)), (xis[0] < MAP['sizex'])), (yis[0] < MAP['sizey']))
    xis=xis[0][indGood]
    yis=yis[0][indGood]
    
    for k,l in zip(xis,yis):
        ex,ey=line(rob_grid_poseX,rob_grid_poseY,k,l)
        MAP['logOdds'][(ex[1:-1]).astype(int),(ey[1:-1]).astype(int)]-= 1
    
    MAP['logOdds'][xis,yis]=MAP['logOdds'][xis,yis]+4
    
    hit=MAP['logOdds']>0
    free=MAP['logOdds']<0
    
    MAP['map'][hit]=1
    MAP['display'][hit]=0
    MAP['display'][free]=1
    MAP['map'] = (MAP['logOdds'] > 0).astype(np.int8)
    
    N_eff= 1 / np.sum(np.square(W))
    if N_eff < N_threshold:
#         print('Resample')
        X = X[np.random.choice(np.arange(N),N,True,W)]
        W = (np.zeros(N))+(1.0/N)
     
    # For intermediate code check
#     if ((i%1000)==0):
#         print(i)
#         MAP['display']= (MAP['logOdds'] < 0).astype(np.int8)
#         fig = plt.figure(figsize=(18,6))
#         ax1 = fig.add_subplot(131)
#         plt.plot(xis,yis,'.k')
#         plt.scatter(rob_grid_poseX,rob_grid_poseY,s=30,c='r')
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.title("Laser reading (red being robot location)")
#         plt.axis('equal')

#         ax2 = fig.add_subplot(132)
#         plt.imshow(MAP['map'],cmap="hot")
#         # pl.plot(coordX,coordY)
#         plt.title('Occupancy Grid')
#         # pl.show()

#         ax3 = fig.add_subplot(133)
#         plt.imshow(scipy.special.expit(MAP['display']) ,cmap="gray")
#         plt.scatter(coordY,coordX,s=1,c='r')
#         plt.show()
    
 


# In[22]:


# Plot Map.
MAP['display']= (MAP['logOdds'] < 0).astype(np.int8)
fig = plt.figure(figsize=(18,6))
ax1 = fig.add_subplot(131)
plt.plot(xis,yis,'.k')
plt.scatter(rob_grid_poseX,rob_grid_poseY,s=30,c='r')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Laser reading (red being robot location)")
plt.axis('equal')

ax2 = fig.add_subplot(132)
plt.imshow(MAP['map'],cmap="hot")
# pl.plot(coordX,coordY)
plt.title('Occupancy Grid')
# pl.show()

ax3 = fig.add_subplot(133)
plt.imshow(1-scipy.special.expit(MAP['logOdds']) ,cmap="gray")
plt.title('Free Space Map')
plt.scatter(coordY,coordX,s=0.1,c='r')
plt.show()

