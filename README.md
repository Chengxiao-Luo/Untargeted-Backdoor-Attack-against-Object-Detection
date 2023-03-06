# Untargeted Backdoor Attack against Object Detection

This is the official implementation of our paper [Untargeted Backdoor Attack against Object Detection](https://www.researchgate.net/publication/365298905_Untargeted_Backdoor_Attack_against_Object_Detection), accepted by the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023. This research project is developed based on Python 3 and Pytorch, created by [Chengxiao Luo](https://github.com/Chengxiao-Luo) and [Yiming Li](http://liyiming.tech/)



## Reference
If our work or this repo is useful for your research, please cite our paper as follows:
```
@inproceedings{luo2023untargeted,
  title={Untargeted Backdoor Attack against Object Detection},
  author={Luo, Chengxiao and Li, Yiming and Jiang, Yong and Xia, Shu-Tao},
  booktitle={ICASSP},
  year={2023}
}
```



## Pipeline
![Pipeline](pipeline.png)



## Requirements

To install requirements:

```setup
pip install -v -e .
pip install -r requirements.txt
```
Make sure the directory follows:
```File Tree
backdoor_attack_against_object_detection
├── configs
│   ├── faster_rcnn
│   ├── sparse_rcnn
│   ├── tood
│   └── ...
├── data
│   ├── coco
│   └── ...
├── mmdet 
│   
├── requirements
│   
├── tools
│   ├── train.py
│   ├── test.py
│   └── ...
|
```


## Dataset Preparation
Download the zipped files of coco dataset and unzip them.

Make sure the directory ``data`` follows:
```File Tree
data
├── coco
│   ├── annotations
│   ├── train2017
│   ├── val2017
│   └── ...
├── ...  
```


>📋  Data Download Link:  
>[train2017](http://images.cocodataset.org/zips/train2017.zip)
>
>[val2017](http://images.cocodataset.org/zips/val2017.zip)
>
>[annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## Train Backdoor Model

Train a backdoor model of Faster-RCNN:
```train
CONFIG_FILE=configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_poisoned_type=1_scale=0.1_rate=0.05_location=center.py
WORK_DIR=logs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_poisoned_type=1_scale=0.1_rate=0.05_location=center

python tools/train.py ${CONFIG_FILE} --gpu-id 0 --work-dir ${WORK_DIR} --seed ${SEED} --auto-scale-lr 
```
## Test Backdoor Model

On Poisoned Datasets:
```Verification
CONFIG_FILE=configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_poisoned_type=1_scale=0.1_rate=0.05_location=center.py
CHECKPOINT_FILE=logs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_poisoned_type=1_scale=0.1_rate=0.05_location=center/latest.pth

python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval bbox --gpu-id 1
```

On Benign Datasets:
```Verification
CONFIG_FILE=configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
CHECKPOINT_FILE=logs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_poisoned_type=1_scale=0.1_rate=0.05_location=center/latest.pth

python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval bbox --gpu-id 1
```

## Acknowledgements

This code is based on [mmdetection](https://github.com/open-mmlab/mmdetection).
