import numpy as np 


def add(x,y):
    return x+y


def body2world(delta_pose):
    '''
    To transform position Coordinates from body to world\\
    coordinates
    variable: delta_pos (Numpy array (x,y,theta))
    '''
    x=delta_pose[0]
    y=delta_pose[1]
    theta=delta_pose[2]
    h=0.93
    a=np.array([[np.cos(theta),-np.sin(theta),0,x],
                [np.sin(theta),np.cos(theta), 0,y],
                [0,            0,             1,h],
                [0,            0,             0,1]])
    return a

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
    print(np.shape(scan.T))

    cos=np.multiply(scan,np.cos(angles))
    sin=np.multiply(scan,np.sin(angles))
    print(np.shape(cos),np.shape(sin))

    z=np.zeros( (1,len(scan.T)) )
    on=np.ones( (1,len(scan.T)) )
    add=np.concatenate((z,on),axis=0)
    add2=np.concatenate((cos,sin),axis=0).T
    print(np.shape(add), np.shape(add2))
    
    
    lid_coord=np.concatenate((add2,add),axis=1)
    
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

# q=Tfm.lidar2body(lid[0],jointAngle['head_angles'][0,b[0]],jointAngle['head_angles'][1,b[0]], indValid)