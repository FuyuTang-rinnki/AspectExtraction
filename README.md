Assignment 3 & 4 - Aspect extraction

Authors: Yifan Song, Jiameng Du, Fuyu Tang

AE - Extracting all aspects of the target entity
```bash
"The hard disk is too noisy."  --> hard disk
"The  retina display is great!" --> retina display
"I like the food, but the service was bad." --> food, service
 ```


# For BERT-PT:

## Environment:

The code is tested on Ubuntu 18.04 with Python 3.6.9(Anaconda), PyTorch 1.3 (apex 0.1) and Transformers 4.5.1.

AWS: Deep Learning AMI (Ubuntu 18.04) Version 42.1 + g4dn.xlarge
```bash
conda create -n p3-torch13 python=3.6.9
conda activate p3-torch13
pip install transformers==4.5.1
pip install scikit-learn
pip install tensorboardX==1.5
conda install pytorch=1.3.0 cudatoolkit=10.0 -c pytorch

git clone https://github.com/NVIDIA/apex
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
## Usage:
To start a new training and save the best_model.pt:  
```bash
rm -rf configs
rm -rf ft_runs
```
```bash
ln -s /usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java java
conda activate p3-torch13
mkdir configs
cp -r reviewlab src/
cp -r reviewlab script/

bash script/run_ft.sh 
```
The valid losses, predictions.json for each run and the f1-score results will be saved in ft_runs/  
To tune the model, you change the value of dropout prob and epsilon in /reviewlab/at_ae.py
Running the code 
```bash
bash script/run_ft.sh
```
agian to see the f1-score, please set TrainConfig.save_best_model = False in script/config.py  
Our best_models can be found at [best models](https://drive.google.com/drive/folders/1gdFyDsIn7nzRoMScmNLMkagq4nJI51qd?usp=sharing)

# For DE-CNN:

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
