# AE-18

## Environment:

The code is tested on Ubuntu 18.04 with Python 3.6.9(Anaconda), PyTorch 1.3 (apex 0.1) and Transformers 2.4.1.

AWS: Deep Learning AMI (Ubuntu 18.04) Version 42.1 + g4dn.xlarge
```bash
conda create -n p3-torch13 python=3.6.9
conda activate p3-torch13
pip install transformers==2.4.1
pip install scikit-learn
pip install tensorboardX==1.5
conda install pytorch=1.3.0 cudatoolkit=10.0 -c pytorch

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Usage:
Running training code (change domain to run different datasets)
```
python script/train.py --domain restaurant
```
Running evaluation code (change domain to run different datasets), current data/official_data/pred.xml file corresponds to the restaurant dataset, and the file data/official_data/pred_laptop.xml corresponds to the laptop dataset. To run evaluation on laptop dataset, need to change filename pred_laptop.xml to pred.xml.
```
python script/evaluation.py --domain restaurant
```


