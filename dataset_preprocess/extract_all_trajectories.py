# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys
import glob

dataset_folder=sys.argv[1]

videos=glob.glob("%s/view_data/*"%dataset_folder)
i=0

for v in videos:
    i+=1

    os.system("python dataset_preprocess/extract_trajectories.py %s %s"%(dataset_folder,v))
   
    
    

   