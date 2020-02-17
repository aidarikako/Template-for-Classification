import os
import argparse
import tensorflow as tf 
import pandas as pd
import cv2
import random
import numpy
from config import train_cfg,val_cfg
from dataloader import MyDataset
from net import MyModel


def adjust_learning_rate(optimizer, epoch):
    optimizer_config = optimizer.get_config()
    lr = optimizer_config['learning_rate']
    if (epoch % train_cfg.epoch_delay==0 and epoch != 0):
        lr *= 0.5 
        optimizer._set_hyper('learning_rate',  lr)
    return lr


def main(args):
    if not os.path.isdir(args.checkpoint):
        try:
            os.makedirs(args.checkpoint)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    #ini log
    log = open(os.path.join(args.checkpoint,'log.txt'),'w')
    log.write('Epoch')
    log.write('\t')
    log.write('LR')
    log.write('\t')
    log.write('Loss')
    log.write('\t')
    log.write('Acc')
    log.write('\t')
    log.write('\n')
    log.flush()

    model = MyModel(classes=train_cfg.num_class)
    # model.build(input_shape=(None,train_cfg.data_shape[0],train_cfg.data_shape[1],3))
    #model._set_inputs((None,train_cfg.data_shape[0],train_cfg.data_shape[1],3))
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4)
    criterion=tf.keras.losses.SparseCategoricalCrossentropy()

    train_data,train_len = MyDataset(train_cfg, train_mode=0).get_dataset()
    val_data,val_len = MyDataset(val_cfg, train_mode=1).get_dataset()

    for epoch in range(args.start_epoch,train_cfg.epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        print('epoch {},lr={}'.format(epoch,lr))
        print('begin the {}th epoch'.format(epoch))
        train_loss=train(train_data,model,criterion,optimizer)
        print('avg_train_loss:{:.5f}'.format(train_loss/train_len))
        acc=eval(val_data,model)
        print('accuracy:{:.5f}'.format(acc))
        log.write('{}'.format(epoch))
        log.write('\t')
        log.write('{:.5f}'.format(lr))
        log.write('\t')
        log.write('{:.5f}'.format(train_loss/val_len))
        log.write('\t')
        log.write('{:.5f}'.format(acc))
        log.write('\t')
        log.write('\n')
        log.flush()
        filename = 'epoch'+str(epoch + 1) + 'checkpoint.h5'
        filepath = os.path.join(args.checkpoint, filename)
        #model.save(filepath=filepath,overwrite=False,include_optimizer=True,save_format='tf')
        model.save_weights(filepath=filepath,overwrite=False,save_format='h5')
    log.close()






def train(train_data,model,criterion,optimizer):
    losses = 0.
    loss_record=0.
    for i,(inputs,targets) in enumerate(train_data):
        with tf.GradientTape() as tape:
            out = model(inputs)
            loss = criterion(targets,out)
        grad = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        losses += loss.numpy()
        loss_record += loss.numpy()
        if(i%10==0 and i!=0):
            print('step {},avg_10_loss:{:.5f}'.format(i,loss_record/10))
            loss_record = 0.
    return losses



def eval(val_data,model):
    data_len=0
    true_len=0
    print('eval......')
    for i,(inputs,targets) in enumerate(val_data):
        out = model(inputs)
        out = out.numpy()
        targets = targets.numpy()
        prediction = numpy.argmax(out, axis=1)
        correct = (prediction == targets).sum()
        data_len += inputs.shape[0]
        true_len += correct
    acc = true_len/data_len
    return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')  
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    main(parser.parse_args())

