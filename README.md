# Shooter_Tracking
Repository for detecting and tracking shooters with gun detection confirmation.

## Overview
<p align="center">
    <img src="/images/ASTERS_CV_Overview.png" width="800">
</p>
The figure above shows an overview of the entire system. We train YOLOv8n using a combination of synthetic and real data. The best model is used for inference with Deep OC-SORT tracking to localize a shooter, enabling a faster and more informed response for law enforcement.

<p align="center">
    <img src="/images/ASTERS_Gun_Confirmation.png" width="800">
</p>
The figure above shows the pipeline of our gun confirmation system for shooter tracking. Shooter and gun detection boxes are sent to Deep OC-SORT. The gun detections are not tracked but are instead used in track initialization to confirm a new shooter detection before assigning that track an ID. After a shooter has an ID, that track no longer requires a gun detection to continue being tracked. If a shooter detection does not match an existing ID and does not have a gun detection to confirm it, it is discarded.

## Dataset
Our dataset,[Real and Synthetic Dataset for Active Shooter Situations](), contains both real and synthetic data and includes annotations for both gun and shooter classes for detection and tracking scenarios. To use with this repository, unzip the downloaded folder and copy the "shooter" folder into the "data" folder and copy "MOT-ASTERS" to the base directory. An overview of the dataset can be seen below.

<p align="center">
    <img src="/images/ASTERS_Dataset_Overview.png" width="800">
</p>

The synthetic data was generated using Unreal Engine 4 and 5 and has two types. The first uses the default, semi-realistic textures of the environments and the second uses segmentation masks as a form of domain randomization. We also augment the semi-realistic textured data with camera sensor effects as another domain adaptation technique. Our data generation process can be seen in the figure below.
<p align="center">
    <img src="/images/ASTERS_Synthetic_Data.png" width="800">
</p>

## Installation
Clone repo and install requirements.txt in a Python>=3.8 enviornment.
~~~
git clone https://github.com/JoshuaRWaite/Shooter_Tracking  # clone
cd Shooter_Tracking
pip install -v -e .
~~~

## Training YOLOv8
Train sequentially using different combinations of data. For example, our *AugCTextured\_CMasked\_Real\_S* model was obtained with the following sequence. Note that the resulting best.pt weights from each run need to be manually renamed and moved to the "weights" folder in the base directory. Note that we include the pretrained weights from our experiments in the weights folder already.
~~~
python train.py --model=yolov8n.pt --data=./settings/shooter_CAT_L.yaml --project=COMB_AUG --name=yolov8n_AugCTextured_L --device=0

python train.py --model=./weights/yolov8n_AugCTextured_L.pt --data=./settings/shooter_CM_L.yaml  --project=COMB_AUG --name=yolov8n_AugCTextured_CMasked --device=0

python train.py --model=./weights/yolov8n_AugCTextured_CMasked.pt --data=./settings/shooter_R_S.yaml --project=COMB_AUG --name=yolov8n_AugCTextured_CMasked_Real_S --device=0
~~~

## Tracking Evaluation
We choose to evaluate the tracking performance of all detection models with Deep OC-SORT with and without our gun detection-based shooter confirmation. To generate the tracking results and plots, run the following lines:
~~~
./tracking_results.sh # tracking with gun confirmation
./tracking_results_shooter.sh # tracking with only shooter detections
python compile_results.py
~~~

## Acknowledgements
This work was supported by the National Science Foundation Award\# CNS-1932505/1932033.

## Citation
TBD

## Paper Links
TBD

## Resources
[YOLOv8](https://github.com/ultralytics/ultralytics)

[YOLO Tracking](https://github.com/mikel-brostrom/yolo_tracking)

[Camera Sensor Effect Augmentation](https://github.com/alexacarlson/SensorEffectAugmentation)
