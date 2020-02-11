

# Template-for-Classification



This is a deep learning template for classification, especially for CPU version, so if you want to train on your GPU, you should change some code. 

The template contains pytorch and tensorflow2. And the code is a template implementation of FlyAi's competition named ["X光片检测患者肺炎''](https://www.flyai.com/d/ChestXray02), so some data preprocessing may not fit all situations.







## Contents

1. [pytorch](#pytorch)
2. [tensorflow2](#tensorflow2)





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

+ Template-for-Classification

   + trainset

      + images

        label.csv

   + valset

      + images

     ​       label.csv

   + testset

      + images

        upload.csv

  + pytorch

  

Activate your anaconda env(like python37),and train your model.

```
python train.py
```

You can change the train config as you like.



#### For testing

Keep updating.......





## tensorflow2

Keep updating.......













