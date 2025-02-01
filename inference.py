# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn
import matplotlib as mpl
import json
mpl.use('Agg')
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch.optim as optim
from simple_vit_1d import Transformer,posemb_sincos_1d

from torch.utils.data import Dataset

import datetime
import json
import argparse
from pathlib import Path
import glob
from scene_to_html import make_vizualizations

from pose_eval import eval_one_seq_ret


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def vec6d_to_R(vector_6D):
    v1=vector_6D[:,:3]/vector_6D[:,:3].norm(dim=-1,keepdim=True)
    v2=vector_6D[:,3:]-(vector_6D[:,3:]*v1).sum(dim=-1,keepdim=True)*v1
    v2=v2/v2.norm(dim=-1,keepdim=True)
    v3=torch.cross(v1,v2,dim=-1)
    return torch.concatenate((v1.unsqueeze(1),v2.unsqueeze(1),v3.unsqueeze(1)),dim=1)

class MyTransformerHead(nn.Module):
    def __init__(self,input_dim,dim,use_positional_encoding_transformer):
        super(MyTransformerHead,self).__init__()
       
        patch_dim=input_dim+1
        self.layers=3
        # dim=128
        self.use_positional_encoding_transformer=use_positional_encoding_transformer
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.transformer_frames=[]
        self.transformer_points=[]
        
        for i in range(self.layers):

            self.transformer_frames.append(Transformer(dim, 1, 16, 64, 2048))
            self.transformer_points.append(Transformer(dim, 1, 16, 64, 2048))
        self.transformer_frames=nn.ModuleList(self.transformer_frames)
        self.transformer_points=nn.ModuleList(self.transformer_points)

    def forward(self, x):
        
        
        x=torch.cat((x,torch.ones(x.shape[0],x.shape[1],1,x.shape[3]).cuda()),dim=2)
        
        x=x.transpose(2,3)
        
        b,n,f,c=x.shape

        x=self.to_patch_embedding(x)
        
        x=x.view(b*n,f,-1) # x.shape [390, 33, 256]
        if self.use_positional_encoding_transformer:
            pe = posemb_sincos_1d(x) #pe.shape= [33,256] (33 frame, 256 embedding dim)
            x=pe.unsqueeze(0)+x 
        for i in range(self.layers):
            
            
            #frames aggregation
            x=self.transformer_frames[i](x)
            

            #point sets aggregation
            x=x.view(b,n,f,-1).transpose(1,2).reshape(b*f,n,-1) 
            
            x=self.transformer_points[i](x)

            x=x.view(b,f,n,-1)
            x=x.transpose(1,2).reshape(b*n,f,-1)

        x=x.view(b,n,f,-1)
        x=x.transpose(2,3)

       
        return x

def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)  
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output


class TracksTo4DNet(nn.Module):
    def __init__(self,width1=320,conv2_kernel_size=31,K=12,conv_kernel_size=3,inputdim=2,use_positionl_encoding=True,positional_dim=4,use_transformer=True,detach_cameras_dynamic=True,use_positional_encoding_transformer=True,use_set_of_sets=False,predict_focal_length=False):
        super(TracksTo4DNet, self).__init__()
        self.predict_focal_length=predict_focal_length
        self.inputdim = inputdim
        self.n1 = width1

        self.K=K
        self.n2 = 6+3+1+self.K+2
        self.detach_cameras_dynamic=detach_cameras_dynamic
        l=conv_kernel_size
        # layers
        self.use_set_of_sets=use_set_of_sets
        self.use_positionl_encoding=use_positionl_encoding
        self.positional_dim=positional_dim
        actual_input_dim=inputdim
        if self.use_positionl_encoding:
            actual_input_dim=2 * inputdim * self.positional_dim+inputdim

        self.use_transformer=use_transformer
       
   
        
        if self.use_positionl_encoding:
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(self.positional_dim)],requires_grad = False)
        
        if True:
            if self.use_transformer:
                self.transformer_my=MyTransformerHead(actual_input_dim,width1,use_positional_encoding_transformer)
          
            

            self.conv_final = nn.Conv1d(self.n1, self.n2, kernel_size=conv2_kernel_size,stride=1, padding=conv2_kernel_size//2, padding_mode='circular')
            
            self.fc1 = nn.Linear(self.n1,3*self.K+1)



            torch.nn.init.xavier_uniform_(self.conv_final.weight)

            torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        
        if self.use_positionl_encoding:
            x_original_shape=x.shape
            x=x.transpose(2,3)
            x=x.reshape(-1,x.shape[-1])

            

            if self.b.device!=x.device:
                self.b=self.b.to(x.device)
            pos = positionalEncoding_vec(x,self.b)
            x=torch.cat((x,pos),dim=1)
            x=x.view(x_original_shape[0],x_original_shape[1],x_original_shape[3],x.shape[-1]).transpose(2,3)



        b = len(x)
        n= x.shape[1]
        l= x.shape[-1]
        if self.use_set_of_sets:
            cameras,perpoint_features=self.set_of_sets_my(x)
        else:
            if  self.use_transformer:
                x=self.transformer_my(x)
            else:
                for i in range(len( self.conv1)):
                    if i==0:
                        x = x.reshape(n*b, x.shape[2],l)
                    else:
                        x = x.view(n * b, self.n1, l)
                    x1 = self.bn1[i](self.conv1[i](x)).view(b,n,self.n1,l)
                    x2 = self.bn1s[i](self.conv1s[i](x)).view(b,n,self.n1,l).mean(dim=1).view(b,1,self.n1,l).repeat(1,n,1,1)
                    x = F.relu(x1 + x2)

            cameras=torch.mean(x,dim=1) 
            cameras=self.conv_final(cameras)
            perpoint_features = torch.mean(x,dim=3)
            perpoint_features = self.fc1(perpoint_features.view(n*b,self.n1))
        B=perpoint_features[:,:self.K*3].view(b,n,3,self.K)
        NR=F.elu(perpoint_features[:,-1].view(b,n))+1+0.00001
 

        
        position_params=cameras[:,:3,:]
        if self.predict_focal_length:
            focal_params=1+0.05*cameras[:,3:4,:].clone().transpose(1,2)
        else:
            focal_params=1.0
        basis_params=cameras[:,4:4+self.K]
        basis_params[:,0,:]=torch.clamp(basis_params[:,0,:].clone(),min=1.0,max=1.0)

        basis_params.transpose(1,2).unsqueeze(1).unsqueeze(1)
        rotation_params=cameras[:,4+self.K:4+self.K+6]

        
        # Converting rotation parameters into a valid rotation matrix (probably better to move to 6d representation)
        rotation_params=vec6d_to_R(rotation_params.transpose(1,2).reshape(b*l,6)).view(b,l,3,3)
        
        # Transfering global 3D into each camera coordinates (using per camera roation and translation)
      
        points3D_static=((basis_params.transpose(1,2).unsqueeze(1).unsqueeze(1))[:,:,:,:,:1]*B.unsqueeze(-2)[:,:,:,:,:1]).sum(-1)
        
        if  self.detach_cameras_dynamic==False:
            points3D=((basis_params.transpose(1,2).unsqueeze(1).unsqueeze(1))[:,:,:,:,1:]*B.unsqueeze(-2)[:,:,:,:,1:]).sum(-1)+points3D_static
        else:
            points3D=((basis_params.transpose(1,2).unsqueeze(1).unsqueeze(1))[:,:,:,:,1:]*B.unsqueeze(-2)[:,:,:,:,1:]).sum(-1)+points3D_static.detach()
        points3D=points3D.transpose(1,3)
        points3D_static=points3D_static.transpose(1,3)
        
        
        
        position_params=position_params.transpose(1,2)
        
        if  self.detach_cameras_dynamic==False:
            points3D_camera=(torch.bmm(rotation_params.view(b*l,3,3).transpose(1,2),points3D.reshape(b*l,3,n)-position_params.reshape(b*l,3).unsqueeze(-1)))
            points3D_camera=points3D_camera.view(b,l,3,n)
        else:
            points3D_camera=(torch.bmm(rotation_params.view(b*l,3,3).transpose(1,2).detach(),points3D.reshape(b*l,3,n)-position_params.detach().reshape(b*l,3).unsqueeze(-1)))
            points3D_camera=points3D_camera.view(b,l,3,n)
        points3D_static_camera=(torch.bmm(rotation_params.view(b*l,3,3).transpose(1,2),points3D_static.reshape(b*l,3,n)-position_params.reshape(b*l,3).unsqueeze(-1)))
        points3D_static_camera=points3D_static_camera.view(b,l,3,n)
      
       
        # Projecting from 3D to 2D
        projections=points3D_camera.clone()

        projections_static=points3D_static_camera.clone()

        
        depths=projections[:,:,2,:]
        depths_static=projections_static[:,:,2,:]

        
       
        projectionx=focal_params*projections[:,:,0,:]/torch.clamp(projections[:,:,2,:].clone(),min=0.01)
        projectiony=focal_params*projections[:,:,1,:]/torch.clamp(projections[:,:,2,:].clone(),min=0.01)


        projectionx_static=focal_params*projections_static[:,:,0,:]/torch.clamp(projections_static[:,:,2,:].clone(),min=0.01)
        projectiony_static=focal_params*projections_static[:,:,1,:]/torch.clamp(projections_static[:,:,2,:].clone(),min=0.01)
        



         
        projections2=torch.cat((projectionx.unsqueeze(2),projectiony.unsqueeze(2)),dim=2)
        projections2_static=torch.cat((projectionx_static.unsqueeze(2),projectiony_static.unsqueeze(2)),dim=2)
        
        return focal_params,projections2,projections2_static,rotation_params,position_params,B,points3D,points3D_static,depths,depths_static,0,basis_params,0,0,points3D_camera,NR

def invert_poses(M):
    M_inv=np.zeros_like(M)
    for i in range(M.shape[0]):
        M_inv[i,:3,:3]=M[i,:3,:3].T
        M_inv[i,:3,3]=-M[i,:3,:3].T@M[i,:3,3]

        
    return M_inv

def calculate_depth_evaluation_metrics(camera_num,GT_depth,depths_,tracks_vis,dynamic_gt_mask):
    with torch.no_grad():
        frames_rel_abs_errors=[]
        frames_rel_abs_errors_dy=[]
        delta_1_25=[]
        delta_1_25_2=[]
        delta_1_25_3=[]

        delta_1_25_dy=[]
        delta_1_25_2_dy=[]
        delta_1_25_3_dy=[]
        align_scales=[]
        for ii in range(camera_num):
            cur_depth_values=depths_[0][ii]
            cur_gt_depth_values=GT_depth[0][ii]
            cur_visual=torch.logical_and(tracks_vis[0][ii],cur_gt_depth_values>0)

            cur_depth_values_dynamic=cur_depth_values[torch.logical_and(cur_visual,dynamic_gt_mask.cuda())]
            cur_gt_depth_values_dynamic=cur_gt_depth_values[torch.logical_and(cur_visual,dynamic_gt_mask.cuda())]
            cur_depth_values=cur_depth_values[cur_visual]
            cur_gt_depth_values=cur_gt_depth_values[cur_visual]

            median_val_gt=cur_gt_depth_values.median()
            median_val_outputs=cur_depth_values.median()

            align_scales.append(median_val_gt/median_val_outputs)
            cur_depth_values_aligned=cur_depth_values*median_val_gt/median_val_outputs

            cur_depth_values_dynamic_aligned=cur_depth_values_dynamic*median_val_gt/median_val_outputs


            frames_rel_abs_errors.append(((cur_depth_values_aligned-cur_gt_depth_values).abs()/cur_gt_depth_values).mean())

            frames_rel_abs_errors_dy.append(((cur_depth_values_dynamic_aligned-cur_gt_depth_values_dynamic).abs()/cur_gt_depth_values_dynamic).mean())

            delta=torch.maximum(cur_depth_values_aligned/cur_gt_depth_values,cur_gt_depth_values/cur_depth_values_aligned)
            
            delta_dy=torch.maximum(cur_depth_values_dynamic_aligned/cur_gt_depth_values_dynamic,cur_gt_depth_values_dynamic/cur_depth_values_dynamic_aligned)
            

            delta_1_25.append(((delta<1.25)*1.0).mean())
            delta_1_25_2.append(((delta<(1.25*1.25))*1.0).mean())
            delta_1_25_3.append(((delta<(1.25*1.25*1.25))*1.0).mean())

            delta_1_25_dy.append(((delta_dy<1.25)*1.0).mean())
            delta_1_25_2_dy.append(((delta_dy<(1.25*1.25))*1.0).mean())
            delta_1_25_3_dy.append(((delta_dy<(1.25*1.25*1.25))*1.0).mean())
            
            

        abs_rel_error=torch.stack(frames_rel_abs_errors).mean()
        delta_1_25_error=torch.stack(delta_1_25).mean()
        delta_1_25_2_error=torch.stack(delta_1_25_2).mean()
        delta_1_25_3_error=torch.stack(delta_1_25_3).mean()


        abs_rel_error_dy=torch.stack(frames_rel_abs_errors_dy).mean()
        delta_1_25_error_dy=torch.stack(delta_1_25_dy).mean()
        delta_1_25_2_error_dy=torch.stack(delta_1_25_2_dy).mean()
        delta_1_25_3_error_dy=torch.stack(delta_1_25_3_dy).mean()
        align_scales=torch.stack(align_scales)
        return align_scales,abs_rel_error,delta_1_25_error,delta_1_25_2_error,delta_1_25_3_error,abs_rel_error_dy,delta_1_25_error_dy,delta_1_25_2_error_dy,delta_1_25_3_error_dy

def evaluate_cameras(dataset_name,camera_num,rotation_params_,translation_params_,GT_poses,logs_folder,epoch,with_BA=False):
    cur_output_poses=torch.zeros(camera_num,4,4).float()+torch.eye(4).unsqueeze(0)
    for i in range(camera_num):
        cur=torch.eye(4)
        cur[:3,:3]=rotation_params_[0][i]
        cur[:3,3]=translation_params_[0][i]
        cur_output_poses[i,:3,:3]= cur[:3,:3]
        cur_output_poses[i,:3,3]= cur[:3,3]
    
    ate, rpe_trans, rpe_rot,fixed_poses=eval_one_seq_ret(invert_poses(GT_poses.numpy()[:,:3,:]), invert_poses(cur_output_poses.numpy()[:,:3,:]))
    
    return ate, rpe_trans, rpe_rot 




def evaluate_model(gt_mask,gt_depth,GT_Ks,projections_,tracks,tracks_vis,depths_,epoch,camera_num,GT_poses,
                   rotation_params_,translation_params_,logs_folder,dataset_name):

    
    tracks=tracks[:,:,:2,:]
    
        
    pixels_reprojection_error=((GT_Ks[0][0,0]*projections_-GT_Ks[0][0,0]*tracks.transpose(1,3))[0].transpose(1,2)**2).sum(dim=-1).sqrt()[tracks_vis[0]].mean()

     
    

    try:
        dynamic_gt_mask=(gt_mask.sum(dim=1).squeeze()>40)
        align_scales,abs_rel_error,delta_1_25_error,delta_1_25_2_error,delta_1_25_3_error,abs_rel_error_dy,delta_1_25_error_dy,delta_1_25_2_error_dy,delta_1_25_3_error_dy= calculate_depth_evaluation_metrics(camera_num,
                                                                                                            gt_depth,depths_,tracks_vis,dynamic_gt_mask)

        
        depth_all=(delta_1_25_error,delta_1_25_2_error,delta_1_25_3_error,abs_rel_error)

            
            
        
        depth_dynamic  = (delta_1_25_error_dy,delta_1_25_2_error_dy,delta_1_25_3_error_dy,abs_rel_error_dy)
    except Exception:  
        # No depth or mask provided, cannot evaluate depth.                                                                                    
        depth_all=(0.0,0.0,0.0,0.0)
        depth_dynamic=(0.0,0.0,0.0,0.0)
       
    print("----------------------")

    with torch.no_grad():
        try:
            ate, rpe_trans, rpe_rot = evaluate_cameras(dataset_name,camera_num,rotation_params_,translation_params_,GT_poses,logs_folder,epoch,False)
        except Exception:
            # probably no GT depth or masks are provided 
            ate=0
            rpe_trans=0
            rpe_rot=0
       
    return ate, rpe_trans, rpe_rot,pixels_reprojection_error,depth_dynamic,depth_all
class CustomDataset(Dataset):
    def __init__(self,max_cameras=50,folder_path="" ,max_points=1000,sample_cameras=False,use_visible_input=True,predict_focal_length=False):
        self.folder_path = folder_path
        self.samples=glob.glob(folder_path+"/*.npz")
        self.use_visible_input=use_visible_input
        self.max_cameras=max_cameras
        self.sample_cameras=sample_cameras
        self.max_points=max_points
        self.max_cameras=max_cameras
        self.predict_focal_length=predict_focal_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_element=np.load(self.samples[idx])

     
        camera_num=np.minimum(self.max_cameras,data_element['pred_tracks'].shape[1])
        if self.sample_cameras:
            camera_num=torch.randint(20,self.max_cameras,(1,))[0].item()#np.minimum(50,synthetic_data['pred_tracks'].shape[1])

            trecklets_num=self.max_points
            start_ind=torch.randint(0,data_element['pred_tracks'].shape[1]-camera_num,(1,))[0].item()
            middle_framee=start_ind+(camera_num/2)
            rel_inds=np.where(np.logical_and(np.abs(data_element['started_at'][0][:,:]-middle_framee)<camera_num,(data_element['pred_visibility'][:,start_ind:(start_ind+camera_num)].sum(axis=1)>10)))[1]
           

            selected_inds=np.zeros(trecklets_num,).astype(np.int64)
            num_actual_tracklets=np.minimum(trecklets_num,rel_inds.shape[0])
            selected_inds[:num_actual_tracklets]=rel_inds[torch.randperm(rel_inds.shape[0])[:trecklets_num]]
            
        else:
            trecklets_num=self.max_points
            start_ind=0
            rel_inds=np.where(np.logical_and(data_element['started_at'][0][:,:]<camera_num,(data_element['pred_visibility'][:,start_ind:(start_ind+camera_num)].sum(axis=1)>10)))[1]
            

            selected_inds=np.zeros(trecklets_num,).astype(np.int64)
            num_actual_tracklets=np.minimum(trecklets_num,rel_inds.shape[0])
            selected_inds[:num_actual_tracklets]=rel_inds[torch.randperm(rel_inds.shape[0])[:trecklets_num]]

        # print(camera_num)
        
        tracks=torch.from_numpy(data_element['pred_tracks']).permute(0,2,3,1)[:,selected_inds,:,start_ind:(start_ind+camera_num)].cuda()
        tracks_vis = torch.from_numpy(data_element['pred_visibility'])[:,start_ind:(start_ind+camera_num),selected_inds].cuda()
        if self.predict_focal_length==False and "Ks_all" in data_element.files:
            # We assume the the GT calibration is given in this case
            GT_Ks=torch.from_numpy(data_element['Ks_all']).float()[start_ind:(start_ind+camera_num),:,:].cuda()
        else:
            # We approximate the calibration in the case. The network predicts a correction.
            GT_Ks=torch.zeros(camera_num,3,3).cuda()
            max_coords=torch.from_numpy(data_element['pred_tracks'])[torch.from_numpy(data_element['pred_visibility'])].max(dim=0)[0]
            generic_focal_lengths=max_coords.mean()
            GT_Ks[:,0,0]=generic_focal_lengths
            GT_Ks[:,1,1]=generic_focal_lengths
            GT_Ks[:,0,2]=max_coords[0]/2.0
            GT_Ks[:,1,2]=max_coords[1]/2.0
            GT_Ks[:,2,2]=1
            

        if "Ms_all" in data_element.files: 
            GT_Rs = torch.from_numpy(data_element['Ms_all']).float().cuda()[start_ind:(start_ind+camera_num),:3,:3]
            GT_ts = torch.from_numpy(data_element['Ms_all']).float().cuda()[start_ind:(start_ind+camera_num),:,3]
            GT_poses=torch.zeros(camera_num,4,4).float()
            for i in range(camera_num):
                cur=torch.eye(4)
                cur[:3,:3]=GT_Rs[i]
                cur[:3,3]=GT_ts[i]
                GT_poses[i,:,:] = torch.inverse(cur)
        else:
            GT_poses=[0]
        if "GT_depth_tracks_all" in data_element.files:
            GT_depth = torch.from_numpy(data_element['GT_depth_tracks_all']).float().cuda()[:,start_ind:(start_ind+camera_num),selected_inds]
        else:
            GT_depth= [0] 
        if "GT_mask_tracks_all" in data_element.files:
            GT_mask = (torch.from_numpy(data_element['GT_mask_tracks_all']).float().cuda()[:,start_ind:(start_ind+camera_num),selected_inds])>0
        else:
            GT_mask= [0] 



        


        
        # Normalize the tracks by the inverse calibration matrix:
        for i in range(camera_num):

            invK=GT_Ks[i].inverse()
            tracks[:,:,0,i]*=invK[0,0]
            tracks[:,:,0,i]+=invK[0,2]
            tracks[:,:,1,i]*=invK[1,1]
            tracks[:,:,1,i]+=invK[1,2]
        dataset_name=self.samples[idx].split("/")[-1][:-4]
        
        if self.use_visible_input:
            tracks=torch.cat((tracks,tracks_vis.transpose(1,2).unsqueeze(2)*1.0),dim=2)
        
        return GT_mask[0],GT_depth[0],tracks[0],0,GT_poses,tracks_vis[0],camera_num,GT_Ks,dataset_name,start_ind,num_actual_tracklets
def get_losses(projections_,tracks,tracks_vis,projections_static_,NR,camera_num,depths_,reprojection_dynamic_coeff,B_,sparsity_dynamic_coeff):

    reprojections=((projections_-tracks.transpose(1,3))**2).sum(dim=2)[tracks_vis].sqrt()
    reprojection_error=reprojections.mean()
    reprojections_static=((projections_static_-tracks.transpose(1,3))**2).sum(dim=2)[tracks_vis].sqrt()

    NR_points=NR.unsqueeze(1).tile((1,camera_num[0],1))[tracks_vis]
    reprojection_error_static=(torch.log(NR_points+(reprojections_static**2)/NR_points)).mean()

    
    
    # We make sure that the depth of the observed tracks is positive
    negative_depth_loss=-torch.clamp(depths_[tracks_vis],max=0).sum()
    
    loss=reprojection_error_static+reprojection_error*reprojection_dynamic_coeff+negative_depth_loss
    sparsity_loss=(B_[:,:,:,1:]/(NR.unsqueeze(-1).unsqueeze(-1).detach())).abs().mean()
    loss+=sparsity_loss*sparsity_dynamic_coeff
    
   

    return loss,sparsity_loss,negative_depth_loss,reprojection_error,reprojection_error_static
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_folder_validation', default="data/pet_test_set/our_data_format_4_validation_rgbd", help="Evaluation folder")
    parser.add_argument('--input_checkpoint_file', default="pretrained_checkpoints/TracksTo4D_pretrained_dogs.pt", help="Checkpoint file")
    parser.add_argument('--input_config_file', default="pretrained_checkpoints/TracksTo4D_pretrained_dogs.json", help="Checkpoint config file")
    
    parser.add_argument('--output_results_file', default="temp.json", help="Quantitative results output file")
    parser.add_argument('--finetunning_iterations', type=int, default=0, help="Number of testtime finetuning iteratioms. If 0 is given, no finetunning is applied.")
    parser.add_argument('--predict_focal_length', type=int, default=0, help="If=1, no GT calibration is used, and the model predicts the focal length.")
    parser.add_argument('--save_visualizations', type=int, default=1, help="If=1, html visualizations are saved.")
    

    torch.cuda.set_device(0)

    opt = parser.parse_args()
    save_visualizations = opt.save_visualizations>0
    predict_focal_length = opt.predict_focal_length>0

    input_config_file=opt.input_config_file
    output_results_file=opt.output_results_file
    input_checkpoint_file=opt.input_checkpoint_file
    
    finetunning_iterations=opt.finetunning_iterations
    dataset_folder_validation=opt.dataset_folder_validation

    
    # Opening JSON file
    f = open(input_config_file)
    loaded_config = json.load(f)
    f.close()
    use_visible_input=loaded_config["use_visible_input"]>0
    positional_dim=loaded_config["positional_dim"]
   
    dataset_folder=loaded_config["dataset_folder"]
    
   
    conv_kernel_size=loaded_config["conv_kernel_size"]
    network_width1=loaded_config["network_width1"]
    K_basis=loaded_config["K_basis"]
    max_cameras=loaded_config["max_cameras"]
    
    
    reprojection_dynamic_coeff=loaded_config["reprojection_dynamic_coeff"]
    sparsity_dynamic_coeff=loaded_config["sparsity_dynamic_coeff"]
    

    
    
    validation_data=CustomDataset(max_cameras,dataset_folder_validation,use_visible_input=use_visible_input,predict_focal_length=predict_focal_length)

    


    logs_folder= "runs/"+datetime.datetime.utcnow().strftime("%m_%d_%Y__%H_%M_%S_%f")+"_"+dataset_folder
    Path(logs_folder).mkdir(parents=True, exist_ok=True)

    opt.logs_folder=logs_folder
    input_dim=2
    if use_visible_input:
        input_dim+=1

    
    with open("%s/conf.json"%logs_folder, "w") as out_file:
        json.dump(vars(opt), out_file)

    validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=False)
    

    
  
    
   
    if True:
        sum_validation_loss=0
        sum_validation_reprojection_loss=0
        num_validation_loss=0
        epoch=0
        ates=[]
        rpe_transes=[]
        rpe_rots=[]

        depth_alls=[]
        depth_dynamics=[]

        model_outputs={}
        dataset_names=[]

        pixels_reprojection_errors=[]

        times=[]
        for gt_mask,gt_depth,tracks,_,GT_poses,tracks_vis,camera_num,GT_Ks,dataset_name,start_ind,trecklets_num  in  validation_dataloader:
            
            net=TracksTo4DNet(inputdim=input_dim,conv_kernel_size=conv_kernel_size,width1=network_width1,positional_dim=positional_dim,K=K_basis,predict_focal_length=predict_focal_length)
            net=net.to("cuda")
            checkpoint=torch.load(input_checkpoint_file)
            net.load_state_dict(checkpoint["model_state_dict"])
            
            
            
            
            dataset_names.append(dataset_name)
            tracks=tracks[:,:trecklets_num[0],:,:]
            tracks_vis=tracks_vis[:,:,:trecklets_num[0]]
            try:
                gt_depth=gt_depth[:,:,:trecklets_num[0]]
            except Exception:
                pass

            try: 
                gt_mask=gt_mask[:,:,:trecklets_num[0]]
            except Exception:
                pass


            if finetunning_iterations>0:
                 
                optimizer = optim.Adam(net.parameters(), lr=0.0001)
                
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                best_loss=1000
                for i in range(finetunning_iterations):
                    net.train()
                    optimizer.zero_grad()
                    focal,projections_,projections_static_,rotation_params_,translation_params_,B_,points3D_,points3D_static_,depths_,depths_static_,projection_before_devide_,basis_params,_,_,points3D_camera,NR=net(tracks)
                    loss,sparsity_loss,negative_depth_loss,reprojection_error,reprojection_error_static=get_losses(projections_,tracks[:,:,:2,:],tracks_vis,projections_static_,NR,camera_num,depths_,reprojection_dynamic_coeff,B_,sparsity_dynamic_coeff)
                
                                    
                    loss.backward()
                    optimizer.step()
                    if i%10==0:
                        print(reprojection_error)
                        # print(focal.mean())
                        print(loss)
                    
                    if loss.item()<best_loss:
                        best_loss=loss.item()
                        torch.save(net.state_dict(),"best_model_temp.pt")
                net.load_state_dict(torch.load("best_model_temp.pt"))
            
            with torch.no_grad():
                net.eval()
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

                
                starter.record()
                Ks_,projections_,projections_static_,rotation_params_,translation_params_,B_,points3D_,points3D_static_,depths_,depths_static_,projection_before_devide_,basis_params,_,_,points3D_camera,NR=net(tracks)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                print(curr_time)
                times.append(curr_time)
                loss,sparsity_loss,negative_depth_loss,reprojection_error,reprojection_error_static=get_losses(projections_,tracks[:,:,:2,:],tracks_vis,projections_static_,NR,camera_num,depths_,reprojection_dynamic_coeff,B_,sparsity_dynamic_coeff)
                    
                print(loss)
                sum_validation_loss+=loss.item()
                sum_validation_reprojection_loss+=reprojection_error.item()
                num_validation_loss+=1.0
                print(dataset_name)
                if True:
                    model_outputs[dataset_name[0]]={}
                    model_outputs[dataset_name[0]]["tracks"]=tracks.cpu()
                    model_outputs[dataset_name[0]]["points3D_static"]=points3D_static_.cpu()
                    model_outputs[dataset_name[0]]["points3D"]=points3D_.cpu()
                    model_outputs[dataset_name[0]]["GT_Ks"]=GT_Ks.cpu()
                    model_outputs[dataset_name[0]]["rotation_params"]=rotation_params_.cpu()
                    model_outputs[dataset_name[0]]["translation_params"]=translation_params_.cpu()
                    
                    
                    
                        


                    ate, rpe_trans, rpe_rot,pixels_reprojection_error,depth_dynamic,depth_all=evaluate_model(gt_mask,gt_depth,GT_Ks[0],projections_,tracks,tracks_vis,depths_,epoch,camera_num,GT_poses[0],
                    rotation_params_,translation_params_,logs_folder,dataset_name)
                

                    pixels_reprojection_errors.append(pixels_reprojection_error.cpu().item())

                
                ates.append(ate)
                rpe_transes.append(rpe_trans)
                rpe_rots.append(rpe_rot)
                depth_alls.append(depth_all)
                depth_dynamics.append(depth_dynamic)
                
        print("-------------")


        print("validation_loss:")
        print(sum_validation_reprojection_loss/num_validation_loss)
        print("-------------------------")
        output_results_to_save={}

        for i in range(len(dataset_names)):
            print("-----------------------")
            print(dataset_names[i])
            output_results_to_save[dataset_names[i][0]]={}

            print("ate  %3.3f | rpe_trans %3.3f| rpe_rot %3.3f"%(ates[i],rpe_transes[i],rpe_rots[i]))

            output_results_to_save[dataset_names[i][0]]['ATE']=ates[i]
            output_results_to_save[dataset_names[i][0]]['RPE_trans']=rpe_transes[i]
            output_results_to_save[dataset_names[i][0]]['RPE_rot']=rpe_rots[i]


            output_results_to_save[dataset_names[i][0]]['rep']=pixels_reprojection_errors[i]
            
            print("ALL:           delta_1.25: %3.3f| delta_1.25^2: %3.3f| delta_1.25^2: %3.3f| abs_rel: %3.3f  "%depth_alls[i])
            output_results_to_save[dataset_names[i][0]]['ALL']=torch.tensor(depth_alls[i]).cpu().numpy()
                             
            print("Dynamic:       delta_1.25: %3.3f| delta_1.25^2: %3.3f| delta_1.25^2: %3.3f| abs_rel: %3.3f "%depth_dynamics[i])
            output_results_to_save[dataset_names[i][0]]['Dynamic']=torch.tensor(depth_dynamics[i]).cpu().numpy()
                    
                  
            
            
        output_results_to_save["mean"]={}
        output_results_to_save["median"]={}

        print("---------------------------------------------")
        print("Mean:")
        print(np.array(ates).mean())
        print(np.array(rpe_transes).mean())
        print(np.array(rpe_rots).mean())
            
        output_results_to_save["mean"]['ATE']=np.array(ates).mean()
        output_results_to_save["mean"]['RPE_trans']=np.array(rpe_transes).mean()
        output_results_to_save["mean"]['RPE_rot']=np.array(rpe_rots).mean()

        output_results_to_save["mean"]['rep']=np.array(pixels_reprojection_errors).mean()
       



        print("Median:")
        print(np.median(np.array(ates)))
        print(np.median(np.array(rpe_transes)))
        print(np.median(np.array(rpe_rots)))

        output_results_to_save["median"]['ATE']=np.median(np.array(ates))
        output_results_to_save["median"]['RPE_trans']=np.median(np.array(rpe_transes))
        output_results_to_save["median"]['RPE_rot']=np.median(np.array(rpe_rots))

        output_results_to_save["median"]['rep']=np.median(np.array(pixels_reprojection_errors).mean())

        print("times")

        print(np.array(times).mean()/1000.0)
        print("All: ")
        print(torch.tensor(depth_alls).mean(dim=0))
        output_results_to_save["mean"]["ALL"]=torch.tensor(depth_alls).mean(dim=0).cpu().numpy()
        output_results_to_save["median"]["ALL"]=torch.tensor(depth_alls).median(dim=0)[0].cpu().numpy()



        print("Dynamic:"  )
        print(torch.tensor(depth_dynamics).mean(dim=0))
        output_results_to_save["mean"]["Dynamic"]=torch.tensor(depth_dynamics).mean(dim=0).cpu().numpy()
        output_results_to_save["median"]["Dynamic"]=torch.tensor(depth_dynamics).median(dim=0)[0].cpu().numpy()
        torch.save(model_outputs,"%s/model_outputs.pt"%logs_folder)
        with open(output_results_file, "w") as out_file:
            json.dump(output_results_to_save, out_file,cls=NumpyEncoder)
        with open("%s/quantitative.json"%logs_folder, "w") as out_file:
            json.dump(output_results_to_save, out_file,cls=NumpyEncoder)
        
        if save_visualizations:
           
            make_vizualizations(model_outputs,dataset_folder_validation,logs_folder)



if __name__ == '__main__':
    main()