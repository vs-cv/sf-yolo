# SF-YOLO [ECCV2024]
This repository contains the code used for our work, **'Source-Free Domain Adaptation for YOLO Object Detection,'** presented at the *ECCV 2024 Workshop on Out-of-Distribution Generalization in Computer Vision Foundation Models*. You can find our paper [here](https://arxiv.org/abs/2409.16538).

Here is an example of using SF-YOLO for the Cityscapes to Foggy Cityscapes scenario.
### SF-YOLO installation

```
conda create --name sf-yolo python=3.11
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
### Download datasets

- Download the Foggy Cityscapes dataset from the [official website.](https://www.cityscapes-dataset.com/)
- Convert the datasets to YOLO format and place them into the ./datasets folder.

### Step 1: Train the Target Augmentation Module
Extract target training data :
```shell
cd TargetAugment_train

python extract_data.py --scenario_name city2foggy --images_folder ../datasets/CityScapesFoggy/yolov5_format/images --image_suffix jpg
```

Then train the Target Augmentation Module :
```
python train.py --scenario_name city2foggy --content_dir data/city2foggy --style_dir data/meanfoggy --vgg pre_trained/vgg16_ori.pth --save_dir models/city2foggy --n_threads=8 --device 0
```

### Step 2:  SF-YOLO adaptation

Download the [Cityscapes source model weights](https://drive.proton.me/urls/3VEG0P1GQR#MAfSdjS57GHI) and place them in the `./source_weights` folder, then run:

```
python train_sf-yolo.py --epochs 60 --batch-size 16 --data foggy_cityscapes.yaml --weights ./source_weights/yolov5l_cityscapes.pt --decoder_path TargetAugment_train/models/city2foggy/decoder_iter_160000.pth --encoder_path TargetAugment_train/pre_trained/vgg16_ori.pth --fc1 TargetAugment_train/models/city2foggy/fc1_iter_160000.pth --fc2 TargetAugment_train/models/city2foggy/fc2_iter_160000.pth --style_add_alpha 0.4 --style_path ./TargetAugment_train/data/meanfoggy/meanfoggy.jpg --SSM_alpha 0.5 --device 0 
```

Other scenarios can be run by following the same steps. All source model weights are available [here](https://drive.proton.me/urls/5WFVDJBDAC#EPs8OZmXtbWq).


### Acknowledgment

Thanks to the creators of [YOLOv5](https://github.com/ultralytics/yolov5), [AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and [LODS](https://github.com/Flashkong/Source-Free-Object-Detection-by-Learning-to-Overlook-Domain-Style) , which this implementation is built upon.

