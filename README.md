# Yolov3_pytorch

## ToDo List

- [ ] Preprocessing on GPU
- [ ] Training pipeline

## Installation
#### Clone and install requirements
    $ git clone https://github.com/SamVadidar/Yolov3_pytorch
    $ python3 -m venv /path/to/new/virtual/environment
    $ source /path/to/new/virtual/environment/bin/activate
    Install dependencies:
    https://pytorch.org/get-started/locally/
    $ cd Yolov3_pytorch
    $ sudo pip install -r requirements.txt

#### Download pretrained weights
    $ cd weights
    $ bash download_weights.sh

#### Download COCO
    $ cd data
    $ bash get_coco_dataset.sh

#### Test
Make sure you have a webcam available and then run test.py
