# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np


import cv2

def to_se3(M):
    M_se3=np.zeros((M.shape[0],4,4))
    for i in range(M.shape[0]):
        M_se3[i,:3,:]=M[i,:3,:]
        M_se3[i,3,3]=1.0
    return M_se3


def invert_poses(M):
    M_inv=np.zeros_like(M)
    for i in range(M.shape[0]):
        M_inv[i,:3,:3]=M[i,:3,:3].T
        M_inv[i,:3,3]=-M[i,:3,:3].T@M[i,:3,3]
  

        
    return M_inv

def find6dofScale_numpy( inputPoints,destPoints,with_scale=False ):
    
    inputPoints_mean=inputPoints.mean(axis=1)[:,np.newaxis]
    inputPoints=inputPoints-inputPoints_mean
    
    destPointsmean=destPoints.mean(axis=1)[:,np.newaxis]
    destPoints=destPoints-destPointsmean

    scaleInput=(np.linalg.norm(inputPoints, axis=0)**2).mean()
   

    inputPoints_=inputPoints
    S=inputPoints_@(destPoints.T)/inputPoints.shape[1]
    U,D,V = np.linalg.svd(S)

    R=V.T@np.diag([1,1,np.linalg.det(V.T@(U.T))])@(U.T)
    if with_scale:
        scale=np.trace(np.diag(D)@np.diag([1,1,np.linalg.det(V.T@(U.T))]))/scaleInput
        
    else:
        scale=1.0


    t=destPointsmean-scale*R@inputPoints_mean

    return scale,R,t


def eval_one_seq_ret(gt_Ms, estimated_Ms):

    aligned_reference=to_se3(invert_poses(gt_Ms))


    traj_est_my=to_se3(invert_poses(estimated_Ms))
    
    scale_numpy,R_numpy,t_numpy=find6dofScale_numpy( (traj_est_my[:,:3,3].T),(aligned_reference[:,:3,3].T),with_scale=True )
    sligned_centers_numpy=scale_numpy*R_numpy@(traj_est_my[:,:3,3].T)+t_numpy

    alined_rotations=np.stack([R_numpy@traj_est_my[i,:3,:3] for i in range(traj_est_my.shape[0])])
    aligned_predictions=np.tile(np.eye(4)[np.newaxis,:,:],(traj_est_my.shape[0],1,1))

    aligned_predictions[:,:3,:3]=alined_rotations
    aligned_predictions[:,:3,3]=sligned_centers_numpy.T

    my_ate=np.sqrt((np.linalg.norm(aligned_predictions[:,:3,3]-aligned_reference[:,:3,3],axis=1)**2).mean())

    relative_poses_predictions=[np.linalg.inv(aligned_predictions[j])@(aligned_predictions[j+1]) for j in range(aligned_reference.shape[0]-1)]
    relative_poses_references=[np.linalg.inv(aligned_reference[j])@(aligned_reference[j+1]) for j in range(aligned_reference.shape[0]-1)]


    relative_pose_errors=np.stack([ np.linalg.inv(relative_poses_references[j])@(relative_poses_predictions[j]) for j in range(aligned_reference.shape[0]-1) ])
    my_rpe_trans=np.sqrt((np.linalg.norm((relative_pose_errors)[:,:3,3],axis=1)**2).mean())

    relative_rot_errors=[np.linalg.norm(cv2.Rodrigues(relative_pose_errors[j,:3,:3])[0])*180/np.pi for j in range(len(relative_pose_errors))  ]
    my_rpe_rot=np.sqrt((np.array(relative_rot_errors)**2).mean())

    print("ATE: ", my_ate)
    print("RTE: ", my_rpe_trans)
    print("RRE: ", my_rpe_rot)

    aligned_cam2world=(aligned_predictions[:,:3,:])
    return my_ate, my_rpe_trans, my_rpe_rot,aligned_cam2world

    
    