# TracksTo4D

This repository contains code, pre-trained models, and test data for TracksTo4D (NeurIPS 2024).

- The **code and test data** are released under the **NSCLv1 license** (NVIDIA OneWay Noncommercial License\_22Mar2022.docx).
- The **pre-trained models** are released under Legal Code - Attribution-NonCommercial 4.0 International - Creative Commons (https://creativecommons.org/licenses/by-nc/4.0/legalcode).

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Installing environment

To set up the environment, run:

```bash
conda create -n tracks_to_4d python=3.8 -y
conda activate tracks_to_4d
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install matplotlib einops imageio opencv-python
pip install imageio[ffmpeg]
```

## Getting Started
Unzip the test folder:
```bash
cd data
unzip pet_test_set.zip
cd ..
```


### Running Inference

To perform inference on the provided pet test set and generate depth and camera metrics along with HTML visualizations, run:

```bash
python inference.py --dataset_folder_validation data/pet_test_set/our_data_format_4_validation_rgbd \
                    --input_checkpoint_file pretrained_checkpoints/TracksTo4D_pretrained_cats_dogs.pt \
                    --input_config_file pretrained_checkpoints/TracksTo4D_pretrained_cats_dogs.json
```

Results are saved in the `runs/` directory.

### Test-Time Fine-Tuning

To enable test-time fine-tuning using our unsupervised losses, run:

```bash
python inference.py --dataset_folder_validation data/pet_test_set/our_data_format_4_validation_rgbd \
                    --input_checkpoint_file pretrained_checkpoints/TracksTo4D_pretrained_cats_dogs.pt \
                    --input_config_file pretrained_checkpoints/TracksTo4D_pretrained_cats_dogs.json \
                    --finetunning_iterations 500
```

You can adjust the number of fine-tuning iterations as needed.

### Training

To train a model using your dataset:

```bash
python train.py --dataset_folder data/pet_test_set/our_data_format_4_validation_rgbd \
                --dataset_folder_validation data/pet_test_set/our_data_format_4_validation_rgbd
```

This example uses test data for demonstration; replace it with your training and validation data.

To continue training from our pre-trained checkpoint, run:

```bash
python train.py --dataset_folder data/pet_test_set/our_data_format_4_validation_rgbd \
                --dataset_folder_validation data/pet_test_set/our_data_format_4_validation_rgbd \
                --continue_trainining 1 \
                --continue_trainining_checkpoint pretrained_checkpoints/TracksTo4D_pretrained_cats_dogs.pt
```

Checkpoints are saved in the `runs/` directory.

## Using Your Own Data

To use your own videos, organize the dataset as follows:

```
dataset_folder
 ├── view_data
 │   ├── video1_name
 │   │   ├── resized_video
 │   │   │   ├── 000.jpg
 │   │   │   ├── 001.jpg
 │   │   │   ├── ...
 │   ├── video2_name
 │   │   ├── resized_video
 │   │   │   ├── 000.jpg
 │   │   │   ├── 001.jpg
 │   │   │   ├── ...
```
where any number of videos can be provided. 
### Preprocessing

Before running preprocessing, download the CoTracker code and checkpoint under `thirdparty/` (see `thirdparty/readme.txt` for details).

Then, run:

```bash
python dataset_preprocess/extract_all_trajectories.py dataset_folder
```

This creates:

- `dataset_folder/videos/`: Visualization of tracking points per video.
- `dataset_folder/trajectories/`: `.npz` files containing tracking data.

If known, include internal calibration in `.npz` files using the field `Ks_all`, formatted as a `(50, 3, 3)` NumPy array (for 50 frames), where `np.load()['Ks_all'][i]` is a `3x3` calibration matrix. Example:

```python
np.load("cop3d_rgbd_test/our_data_format_4_validation_rgbd/2024-03-03--16-50-13_260_st_1.npz")['Ks_all']
```

If calibration is not provided, our code approximates the calibration, and the network can apply a correction if --predict_focal_length 1 (inference.py or train.py)

For evaluation, you can also include:

- `GT_mask_tracks_all`: Ground truth mask (1 for foreground objects).
- `GT_depth_tracks_all`: Ground truth depth values.
- `Ms_all`: Ground truth camera poses (world-to-camera transformation matrices).

The `inference.py` script evaluates depth and camera predictions against any provided ground truth.

## Citation

If you find our work useful in your research, please cite:

```bibtex
@inproceedings{kastenfast,
  title={Fast Encoder-Based 3D from Casual Videos via Point Track Processing},
  author={Kasten, Yoni and Lu, Wuyue and Maron, Haggai},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

