# tracks_to_4d

Installing our tested environment:
```
conda create -n tracks_to_4d python=3.8 -y
conda activate tracks_to_4d
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install matplotlib einops imageio opencv-python


```


## Getting started

For installing dependencies:



For inference, run :
```
python inference.py --dataset_folder_validation cop3d_rgbd_test/our_data_format_4_validation_rgbd --input_checkpoint_file pretrained_checkpoints/pets_checkpoint.pt --input_config_file pretrained_checkpoints/conf.json
```
This script runs evaluation on our provided pet test set, outputting depth and camera metrics. It also creates html visualizations. All is saved into runs/.

Inference further allows test time finetunning using the unsupervised lossses. 
This is possible by (you can reduce/increase the number of finetunning iterations):

```
python inference.py --dataset_folder_validation cop3d_rgbd_test/our_data_format_4_validation_rgbd --input_checkpoint_file pretrained_checkpoints/pets_checkpoint.pt --input_config_file pretrained_checkpoints/conf.json --finetunning_iterations 500
```

For training run:

```
python train.py --dataset_folder cop3d_rgbd_test/our_data_format_4_validation_rgbd  --dataset_folder_validation cop3d_rgbd_test/our_data_format_4_validation_rgbd 
```
where this line apply training on the test data, just for providing a concrete example. You should replace this with your training and validation data. 

It is possible to continue the training from our pretrained checkpoint by:
```
python train.py --dataset_folder cop3d_rgbd_test/our_data_format_4_validation_rgbd  --dataset_folder_validation cop3d_rgbd_test/our_data_format_4_validation_rgbd  --continue_trainining 1 --continue_trainining_checkpoint pretrained_checkpoints/pets_checkpoint.pt
```
The checkpoints will be saved to the folder "run/".

## Using your own data

For creating a dataset of your data you should provide video frames in the following way (see ):

```
 dataset_folder
 ---- view_data
 -------- video1_name
 ------------ resized_video
 ---------------- 000.jpg
 ---------------- 001.jpg
 ---------------- 002.jpg
 ...
 --------- video2_name
 ------------ resized_video
 ---------------- 000.jpg
 ---------------- 001.jpg
 ---------------- 002.jpg
 ...
 

```
where any number of videos can be provided. 


Before running the preprocessing script you should download cotracker code and checkpoint under thirdparty.
See more details in thirdparty/readme.txt.

Then, to run the preprocess script: 
```
python dataset_preprocess/extract_all_trajectories.py dataset_folder
```

This creates two folders inside dataset_folder: trajectories, videos. 
dataset_folder/videos contains a visualization for the traking points of each video.
dataset_folder/trajectories contains an npz file for each video, each contains the tracking info.
Note that if the internal calibration information is known, you should add to each npz file a field named: 'Ks_all', that contains a numpy array of size (50, 3, 3) (in case of 50 frames), such that np.load['Ks_all'][i] is a 3 by 3 calibration matrix. 
For example:
```
np.load("cop3d_rgbd_test/our_data_format_4_validation_rgbd/2024-03-03--16-50-13_260_st_1.npz")['Ks_all']
```
Without calibration, our code approximates the calibration, and the network can apply a correction if --predict_focal_length 1 (inference.py or train.py)

For evaluation you can also add: 
'GT_mask_tracks_all', 'GT_depth_tracks_all', 'Ms_all', where 'Ms_all' contains the GT camera poses (world to camera), and 'GT_mask_tracks_all', 'GT_depth_tracks_all' contain the GT mask value (1 represents a point on the foreground object) and depth values. 
The file inference.py runs the evaluate each part (cameras or depth) for any provided GT data. 