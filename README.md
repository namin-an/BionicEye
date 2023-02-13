Project
===============


Step-by-step requirements
-----

1. Clone the repository
```
git clone https://github.com/namin-an/Project.git   
cd Project
```

2. Create the same environment as ours and install dependencies
- For Window users
```
conda env create -n bioniceye --file environment_windows_python39.yaml   
conda activate bioniceye
```
- For Mac users
```
conda env create -n bioniceye --file environment_mac_namina.yaml   
conda activate bioniceye
```
- For any users
```
conda create -n bioniceye python=3.9
conda activate bioniceye
pip install -r requirements.txt
```

3. Download K-Face data from [AI Hub](https://www.aihub.or.kr/) and preprocess them using the below jupyter notebook:
```
selectTrainableImgs.ipynb # Select 4,972 images/facial class that are trainable.   
processBeforeDownsamp.ipynb # Crop the images into squares, remove noisy backgrounds, and enhance the contrast of the images. 
downsampling.ipynb # Make low-resolution phosphene images. 
```

4. Place the original high-resolution data in `data/Middle_Resolution_137_unzipped_parcropped_128_removed_train` and low-resolution phosphene data in `data/sample_for_dev_test`. All the other human experiment data can be found in `data` folder.


5. Change the hyperparameter settings in `code/conf/config.yaml` and run the following scripts:
```
cd code   
```
- To train RL models
```
python3 ../main.py 
```
- To train SL models
```
python ../main.py -m supervised.training.train_num=128,256,384,512,640,768,896,1024,1152,1280,1408,1536,166
```


Code implementations
-----
Image processing codes are from [namin-an/ArtificialVision](https://github.com/namin-an/ArtificialVision)   
Reinforcement learning algorithm codes are adapted from [seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL)   



Device specifications
-----

`Processor` Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz

`Operating system` Windows 10 Pro 64 bit

`Memory` 16.0GB

`SSD` Samsung SSD 970 EVO Plus 500GB

`Graphics` NVIDIA GeForce RTX 3070   


Citation
-----
If you want to report the results of our method or implement the framework, please cite as follows:   
```
@INPROCEEDINGS{?,
  author    = {An, Na Min and Roh, Hyeonhee and Kim, Sein and Kim, Jae Hun and Im, Maesoon},
  booktitle={[Proceedings 2023] IJCNN International Joint Conference on Neural Networks}, 
  title     = {Reinforcement Learning-Based Feedback Framework to Simulate Human Psychophysical Experiment of Prosthetic Vision},
  year      = {2023},
  volume    = {?},
  pages     = {1-8},
  doi       = {?}
}
```