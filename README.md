# Lane-Detection-Based-on-CycleGAN-and-Feature-Fusion-in-Challenging-Scenes


## framework
![images](https://github.com/Chris871029nose/Lane-Detection-Based-on-CycleGAN-and-Feature-Fusion-in-Challenging-Scenes/blob/main/data/framework.png)
## Datasets
CULane  
The whole dataset is available at [CULane](https://xingangpan.github.io/projects/CULane.html).
```
CULane
├── driver_23_30frame       # training&validation
├── driver_161_90frame      # training&validation
├── driver_182_30frame      # training&validation
├── driver_193_90frame      # testing
├── driver_100_30frame      # testing
├── driver_37_30frame       # testing
├── laneseg_label_w16       # labels
└── list                    # list
```
## Getting Started
```
conda create -n  your_env_name python=3.6
conda activate your_env_name
conda install pytorch==1.3.0 torchvision==0.4.1 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt 
```
# GAN Model
## training
```
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
## testing
```
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
# Lane Detection
## training
```
sh train_erfnet.sh
```
## testing
1. Run test script
```
sh test_erfnet.sh
```
2. Get lines from probability maps
```
cd tools/prob2lines
matlab -nodisplay -r "main;exit"
```
Please check the file path in Matlab code before.

3. Evaluation
```
cd /tools/lane_evaluation
make
# You may also use cmake instead of make, via:
# mkdir build && cd build && cmake ..
sh eval_all.sh    # evaluate the whole test set
sh eval_split.sh  # evaluate each scenario separately
```
# Example
![images](https://github.com/Chris871029nose/Lane-Detection-Based-on-CycleGAN-and-Feature-Fusion-in-Challenging-Scenes/blob/main/data/result.png)
# Acknowledgement
This project refers to the following projects:

* [SCNN](https://github.com/XingangPan/SCNN)
* [Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)
* [DDRNet](https://github.com/ydhongHIT/DDRNet)
* [dla](https://github.com/ucbdrive/dla)
* [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [Light-Condition-Style-Transfer](https://github.com/Chenzhaowei13/Light-Condition-Style-Transfer?tab=readme-ov-file#acknowledgement)
