

# Template-for-Classification



This is a deep learning template for classification, especially for CPU version, so if you want to train on your GPU, you should change some code. 

The template contains pytorch and tensorflow2. And the code is a template implementation of FlyAi's competition named ["X光片检测患者肺炎''](https://www.flyai.com/d/ChestXray02), so some data preprocessing may not fit all situations.



## Contents

1. [note](#note)
2. [pytorch](#pytorch)
3. [tensorflow2](#tensorflow2)



## note

**Networks**: I use resnet as the networks' backbone. And I remove the last two layers, connect with a globalNet which use to fuse multi-scale features. 

**Data preprocessing**: Because of the dataset's feature  of ["X光片检测患者肺炎''](https://www.flyai.com/d/ChestXray02), so I only do random rotation([-15<sup>o</sup>,15<sup>o</sup>]) for the data augmentation.



## pytorch

#### For training

Clone the code from github and cd the pytorch folder.

```
git clone https://github.com/aidarikako/Template-for-Classification.git
```

```
cd pytorch
```

Import  python37.yml to your own anaconda env. 

Make your dir as follow:

```
${Template-for-Classification}
|-- trainset
|       |-- images
|       |       |--xxxxx.png
|       |       |--...
|   `   |-- label.csv
|
|-- valset
|       |-- images
|       |       |--xxxxx.png
|       |       |--...
|   `   |-- label.csv
|
|-- testset
|       |-- images
|       |       |--xxxxx.png
|       |       |--...
|   `   |-- upload.csv
|
|-- pytorch

```



Activate your anaconda env(like python37), and train your model.

```
python train.py
```

You can change the train config as you like.



#### For testing

After you train your model, go to the checkpoint folder,and find the checkpoint name. 

You can use your model to test  as follow:

```
python test.py -c 'your checkpoint's name'
```

You can change the test config as you like and your prediction results will save at  upload.csv.



## tensorflow2

Keep updating.......













