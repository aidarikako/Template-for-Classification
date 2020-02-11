import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from dataloader import MyDataset
from config import test_cfg
import pandas as pd
from net import networks
import numpy

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=networks(num_class=test_cfg.num_class, pretrained = False).to(device)
    test_loader = torch.utils.data.DataLoader(
        MyDataset(test_cfg,train_mode=2),
        batch_size=test_cfg.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    predict_np=[]
    checkpoint_file = os.path.join(test_cfg.checkpoint_path,args.checkpoint +'.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print('successful loaded checkpoint:{} (epoch{})'.format(checkpoint_file,checkpoint['epoch']))

    model.eval()
    
    print('testing.........')
    for i, inputs in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs)
            out = model(input_var)
            _, prediction = torch.max(out.data, 1)
            pre = prediction.numpy()
            predict_np.append(pre)

    csv_path = os.path.join(test_cfg.img_folder,'..',test_cfg.csv_name)
    data = pd.read_csv(csv_path)
    predict_np=numpy.concatenate(predict_np)
    data['labels'] = pd.DataFrame(predict_np)
    data.to_csv(csv_path,index=False)
    print('successful write the predict results!')
    
       
            









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')  
    parser.add_argument('-c', '--checkpoint', default='epoch1checkpoint', type=str, metavar='PATH',
                        help='checkpoint filename (default: epoch1checkpoint)')
    main(parser.parse_args())
