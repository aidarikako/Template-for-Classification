#cpu version

import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader import MyDataset
from config import train_cfg,val_cfg
from net import networks


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if (epoch % train_cfg.epoch_delay==0 and epoch != 0):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
    return optimizer.state_dict()['param_groups'][0]['lr']




def main(args):
    if not os.path.isdir(args.checkpoint):
        try:
            os.makedirs(args.checkpoint)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=networks(num_class=train_cfg.num_class, pretrained = True).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    

    train_loader = torch.utils.data.DataLoader(
        MyDataset(train_cfg,train_mode=0),
        batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        MyDataset(val_cfg,train_mode=1),
        batch_size=val_cfg.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    #ini log
    log = open(os.path.join(args.checkpoint,'log.txt'),'w')
    log.write('Epoch')
    log.write('\t')
    log.write('LR')
    log.write('\t')
    log.write('Avg Train Loss')
    log.write('\t')
    log.write('Acc')
    log.write('\t')
    log.write('\n')
    log.flush()


    print('Total params: %.2fMB' % (sum(p.numel() for p in model.parameters())/(1024*1024)*4))

    for epoch in range(args.start_epoch,train_cfg.epochs):
        lr=adjust_learning_rate(optimizer, epoch)
        print('epoch {},lr={}'.format(epoch,lr))
        print('begin the {}th epoch'.format(epoch))
        train_loss=train(train_loader,model,criterion,optimizer)
        print('train_loss:{:.5f}'.format(train_loss))
        acc=eval(eval_loader,model)
        print('accuracy:{:.5f}'.format(acc))
        log.write('{}'.format(epoch))
        log.write('\t')
        log.write('{:.5f}'.format(lr))
        log.write('\t')
        log.write('{:.5f}'.format(train_loss/train_loader.__len__()))
        log.write('\t')
        log.write('{:.5f}'.format(acc))
        log.write('\t')
        log.write('\n')
        log.flush()
        filename = 'epoch'+str(epoch + 1) + 'checkpoint.pth.tar'
        filepath = os.path.join(args.checkpoint, filename)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, filepath)

    log.close()


def train(train_loader,model,criterion,optimizer):
    model.train()
    losses = 0.
    loss_record=0.
    for i,(inputs,targets) in enumerate(train_loader):
        input_var = torch.autograd.Variable(inputs).float()
        target_var = torch.autograd.Variable(targets)
        out = model(input_var)
        loss = criterion(out,target_var)
        loss_record += loss.data.item()
        losses += loss.data.item()
        if(i%10==0 and i!=0):
            print('step {},10_loss_loss:{:.5f}'.format(i,loss_record/10))
            loss_record = 0.            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses

def eval(eval_loader,model):
    model.eval()
    print('eval......')
    data_len=0
    true_len=0
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(eval_loader)):
            input_var = torch.autograd.Variable(inputs)
            out = model(input_var)
            _, prediction = torch.max(out.data, 1)
            correct = (prediction == targets).sum().item()
            data_len += inputs.shape[0]
            true_len += correct
        acc = true_len/data_len
    return acc
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')  
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    main(parser.parse_args())