# Projective Subspace Networks


## Dependencies
* Pytorch 0.4+
* TorchVision / cv2
* numpy
* pickle
* python 3.5+
* tqdm



#Data Download : 

### miniImageNet
[[Google Drive](https://drive.google.com/open?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY)]

Put downloaded data in specified data path, in my case : /home/csimon/research/data/miniimagenet/split/
Just put the data files (*.pkl): './data/miniimagenet/split'

### PSN : 

### 5-way 5-shot Train
`python3 train_psn_main.py --shot 5 --train-way 5 --save-path ./save/psn-model --data-path ./data/miniimagenet/split`

### 5-way 5-shot Test
`python3 test_psn.py --load ./save/psn-model/max-acc.pth --shot 5 --way 5`





