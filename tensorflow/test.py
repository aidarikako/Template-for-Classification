import os
import argparse
import tensorflow as tf 
import pandas as pd
import cv2
import numpy
from tqdm import tqdm
from config import test_cfg
from dataloader import MyDataset
from net import MyModel

def main(args):

    model = MyModel(classes=test_cfg.num_class)

    test_data,test_len = MyDataset(test_cfg, train_mode=2).get_dataset()


    predict_np=[]
    checkpoint_file = os.path.join(test_cfg.checkpoint_path,args.checkpoint +'.h5')
    model.load_weights(checkpoint_file)
    print('successful loaded checkpoint:{} (epoch{})'.format(checkpoint_file,args.checkpoint[5]))  

    print('testing.........')
    for i, (inputs,targets) in tqdm(enumerate(test_data)):
        out = model(inputs)
        out = out.numpy()
        prediction = numpy.argmax(out, axis=1)
        predict_np.append(prediction)

    csv_path = os.path.join(test_cfg.img_folder,'..',test_cfg.csv_name)
    data = pd.read_csv(csv_path)
    predict_np=numpy.concatenate(predict_np)
    data['labels'] = pd.DataFrame(predict_np)
    data.to_csv(csv_path,index=False)
    print('successful write the predict results!')
    
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Testing')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')  
    parser.add_argument('-c', '--checkpoint', default='epoch1checkpoint', type=str, metavar='PATH',
                        help='checkpoint filename (default: epoch1checkpoint)')
    main(parser.parse_args())
