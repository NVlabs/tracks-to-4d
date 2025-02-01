# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import torch
import glob
from base64 import b64encode
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../thirdparty/co-tracker'))
from cotracker.utils.visualizer import Visualizer, read_video_from_path
import imageio
import numpy as np
dataset_folder=sys.argv[1]
video_name=sys.argv[2]
print("video_name")
print(video_name)

framess=[]
for i in range(0,len(glob.glob('%s/resized_video/*.jpg'%video_name)),1):
    im=imageio.imread("%s/resized_video/%03d.jpg"%(video_name,i))
    framess.append(im)

video=torch.from_numpy(np.stack(framess)).permute(0,3,1,2).unsqueeze(0).float()

from cotracker.predictor import CoTrackerPredictor
torch.cuda.set_device(1)
model = CoTrackerPredictor(
    checkpoint=os.path.join(
        './thirdparty/co-tracker/checkpoints/cotracker2.pth'
    )
)

if torch.cuda.is_available():
    model = model.cuda()
    video = video.cuda()

max_grid_query_frame=len(framess)-1
all_pred_tracks=[]
all_pred_visibility=[]
started_at=[]
for query_frame in range(0,max_grid_query_frame,20):
    print(query_frame)
    pred_tracks, pred_visibility = model(video, grid_size=15,grid_query_frame=query_frame,backward_tracking=True)
    started_at.append(query_frame*torch.ones(1,1,pred_tracks.shape[2]))
    all_pred_tracks.append(pred_tracks)
    all_pred_visibility.append(pred_visibility)

query_frame=max_grid_query_frame
pred_tracks, pred_visibility = model(video, grid_size=15,grid_query_frame=query_frame,backward_tracking=True)
started_at.append(query_frame*torch.ones(1,1,pred_tracks.shape[2]))
all_pred_tracks.append(pred_tracks)
all_pred_visibility.append(pred_visibility)

pred_visibility=torch.cat(all_pred_visibility,dim=2)
pred_tracks=torch.cat(all_pred_tracks,dim=2)
started_at=torch.cat(started_at,dim=2)
from pathlib import Path
Path("%s/videos"%dataset_folder).mkdir(parents=True, exist_ok=True)
Path("%s/trajectories"%dataset_folder).mkdir(parents=True, exist_ok=True)
vis = Visualizer(save_dir='%s/videos'%dataset_folder, pad_value=100)

vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename=video_name.split("/")[-1])
mapp={"pred_tracks":pred_tracks.cpu().numpy(),"pred_visibility":pred_visibility.cpu().numpy(),"started_at":started_at.cpu().numpy()}
import numpy as np
np.savez('%s/trajectories/%s.npz'%(dataset_folder,video_name.split("/")[-1]),**mapp)