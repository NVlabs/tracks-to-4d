from inference import TracksTo4DNet,CustomDataset,get_losses
import argparse
import torch
import torch.optim as optim
import datetime
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import json 
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sparsity_dynamic_coeff', type=float, default=0.001, help="Sparsity loss coefficient")
    parser.add_argument('--reprojection_dynamic_coeff', type=float, default=50.0, help="Reprojection error coefficient")
    parser.add_argument('--dataset_folder', default="data/pet_test_set/our_data_format_4_validation_rgbd", help="Training data path")
    
    parser.add_argument('--dataset_folder_validation', default="data/pet_test_set/our_data_format_4_validation_rgbd", help="Validation data path")
    
    # Architechture hyper parameters
    parser.add_argument('--conv_kernel_size', type=int, default=15)
    parser.add_argument('--network_width1', type=int, default=256)
    parser.add_argument('--positional_dim', type=int, default=12, help="if only_static>0, only optimize the static part")
    parser.add_argument('--K_basis', type=int, default=12, help="if only_static>0, only optimize the static part")
    


    parser.add_argument('--max_cameras', type=int, default=50, help="Maximum number of frames")
    

    parser.add_argument('--continue_trainining_checkpoint', default="pretrained_checkpoints/TracksTo4D_pretrained_cats.pt", help="Path of the checkpoint to continue training from. (Used if continue_trainining>0)")
    parser.add_argument('--continue_trainining', type=int, default=0, help="if >0, continue training from a checkpoint")
    
    
    parser.add_argument('--predict_focal_length', type=int, default=0, help="If>0, the network predicts focal length correction.")
    
   
    

    opt = parser.parse_args()

    predict_focal_length=opt.predict_focal_length>0
    positional_dim=opt.positional_dim
    continue_trainining=opt.continue_trainining>0
    continue_trainining_checkpoint=opt.continue_trainining_checkpoint
    dataset_folder=opt.dataset_folder
    dataset_folder_validation=opt.dataset_folder_validation

    conv_kernel_size=opt.conv_kernel_size
    network_width1=opt.network_width1

    K_basis=opt.K_basis
    max_cameras=opt.max_cameras
    training_data_init=CustomDataset(22,dataset_folder,max_points=100,sample_cameras=False)
    training_data=CustomDataset(max_cameras,dataset_folder,max_points=100,sample_cameras=True)
    
    validation_data=CustomDataset(max_cameras,dataset_folder_validation,max_points=100)

    reprojection_dynamic_coeff=opt.reprojection_dynamic_coeff

    sparsity_dynamic_coeff=opt.sparsity_dynamic_coeff
    
    if continue_trainining:

        logs_folder=continue_trainining_checkpoint[:-22]
    else:
        logs_folder= "runs/"+datetime.datetime.utcnow().strftime("%m_%d_%Y__%H_%M_%S_%f")+"_"+dataset_folder
    checkpoints_folder="%s/checkpoints"%logs_folder
    Path(logs_folder).mkdir(parents=True, exist_ok=True)
    Path(checkpoints_folder).mkdir(parents=True, exist_ok=True)
    opt.logs_folder=logs_folder
    input_dim=3
    
    net=TracksTo4DNet(inputdim=input_dim,conv_kernel_size=conv_kernel_size,width1=network_width1,positional_dim=positional_dim,K=K_basis,predict_focal_length=predict_focal_length)
    net=net.to("cuda")
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    
    num_pre_train_epochs=100

    with open("%s/conf.json"%logs_folder, "w") as out_file:
        json.dump(vars(opt), out_file)

    train_dataloader = DataLoader(training_data_init, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)
    
    start_epoch=0
    if continue_trainining:
        checkpoint=torch.load(continue_trainining_checkpoint)
        aa=0
        start_epoch=checkpoint["epoch"]+1
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    
    if True:
        if start_epoch>=50:
                train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
        for epoch in range(start_epoch,1000000,1):
            if epoch==50:
                train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            
            net.train()
            sum_epoch_loss=0
            sum_epoch_reprojection_loss=0
            num_epoch_tot=0

            for gt_mask,gt_depth,tracks,_,GT_poses,tracks_vis,camera_num,GT_Ks,dataset_name,start_ind,trecklets_num  in  train_dataloader:
                optimizer.zero_grad()
                if trecklets_num.min()<100:
                    continue
                focal,projections_,projections_static_,rotation_params_,translation_params_,B_,points3D_,points3D_static_,depths_,depths_static_,projection_before_devide_,basis_params,_,_,points3D_camera,NR=net(tracks)
               
                if epoch<num_pre_train_epochs:
                    if torch.isnan(translation_params_.sum()):
                        aaa=0
                    loss=((translation_params_-torch.tensor([[[0.0,0.0,-15.0]]]).cuda())**2).sum(dim=1).mean()
                    loss+=((rotation_params_-torch.eye(3).unsqueeze(0).unsqueeze(0).cuda())**2).mean()*100
                    if torch.isnan(loss):
                        aaa=0
                    loss/=100

                    if loss<0.0001:
                        num_pre_train_epochs=0

                    print("epoch %d"%epoch)
                    print(loss)
                else:
                    loss,sparsity_loss,negative_depth_loss,reprojection_error,reprojection_error_static=get_losses(projections_,tracks[:,:,:2,:],tracks_vis,projections_static_,NR,camera_num,depths_,reprojection_dynamic_coeff,B_,sparsity_dynamic_coeff)
                    if epoch>3:
                        loss=torch.clamp(loss,max=100)
                    sum_epoch_reprojection_loss += reprojection_error.item()
                
                loss.backward()
                optimizer.step()
                sum_epoch_loss+=loss.item()
                
                num_epoch_tot+=1
                
            
            mean_epoch_loss=sum_epoch_loss/num_epoch_tot
            if epoch%10==0:
                print("---epoch---%d"%epoch)
                print(mean_epoch_loss)
           

            if (epoch%100==0  ) and epoch>0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_epoch_loss
                }, "%s/%06d.pt"%(checkpoints_folder,epoch))
                net.eval()
              
                
                with torch.no_grad():
                    sum_validation_loss=0
                    sum_validation_reprojection_loss=0
                    num_validation_loss=0
                    
                    for gt_mask,gt_depth,tracks,_,GT_poses,tracks_vis,camera_num,GT_Ks,dataset_name,start_ind,trecklets_num  in  validation_dataloader:
                        focal,projections_,projections_static_,rotation_params_,translation_params_,B_,points3D_,points3D_static_,depths_,depths_static_,projection_before_devide_,basis_params,_,_,points3D_camera,NR=net(tracks)
                        loss,sparsity_loss,negative_depth_loss,reprojection_error,reprojection_error_static=get_losses(projections_,tracks[:,:,:2,:],tracks_vis,projections_static_,NR,camera_num,depths_,reprojection_dynamic_coeff,B_,sparsity_dynamic_coeff)
                    
                        sum_validation_loss+=loss.item()
                        sum_validation_reprojection_loss+=reprojection_error.item()
                        num_validation_loss+=1.0
                    
                    print("-------------------------")
                    print("validation_loss:")
                    print(sum_validation_loss/num_validation_loss)
                    print("-------------------------")

                    

if __name__ == '__main__':
    main()