# Yolov3

## Installation
#### Clone and install requirements
    $ git clone https://github.com/SamVadidar/Yolov3
    $ python3 -m venv /path/to/new/virtual/environment
    $ source /path/to/new/virtual/environment/bin/activate
    $ sudo pip3 install -r requirements.txt
    $ cd Yolov3

#### Download pretrained weights
    $ cd weights
    $ bash download_weights.sh

#### Download COCO
    $ cd data
    $ bash get_coco_dataset.sh