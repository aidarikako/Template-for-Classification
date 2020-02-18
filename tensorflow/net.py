import tensorflow as tf
from resnet import *
#from globalnet import globalNet



class MyModel(tf.keras.Model):
    def __init__(self,classes=1):
        super(MyModel,self).__init__()
        #self.resnet = ResNet50(classes=classes,include_top=True,weights=None)
        self.resnet = ResNet50(classes=4,include_top=True,weights='imagenet')
    
    def call(self,x):
        x=self.resnet(x)       
        return x
 