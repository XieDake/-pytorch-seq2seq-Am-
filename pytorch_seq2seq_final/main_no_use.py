# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
            seq2Seq的main程序！
===============================================================
"""
import argparse,math,os
import torch
from config import Config
from data_helper import sent2id,batch_yeild
from models import Seq2seq_all_for_batch,Encoder_for_batch,Decoder_Luong_AM_for_batch
from train import train_one_epoch,validation_after_one_epoch,train_one_epoch_0601

def parse_arguments():
    parse = argparse.ArgumentParser(description='Hyperparams of this project!')
    #
    parse.add_argument('--encoder_hidden_dim', type=int, default=256,help='Hidden dim of encoder!')
    parse.add_argument('--encoder_embed_dim', type=int, default=256,help='Embed dim of encoder!')
    parse.add_argument('--encoder_num_layers', type=int, default=1,help='Num layers of encoder!')

    parse.add_argument('--decoder_hidden_dim', type=int, default=256,help='Hidden dim of decoder!')
    parse.add_argument('--decoder_embed_dim', type=int, default=256,help='Embed dim of decoder!')
    parse.add_argument('--decoder_num_layers', type=int, default=1,help='Num layers of decoder!')
    parse.add_argument('--decoder_droupOut_p',type=float,default=0.5,help='DroupOut prob of decoder!')
    parse.add_argument('--decoder_method',type=str,default='general',help='Align methods of AM in decoder!')
    #
    parse.add_argument('--epochs', type=int, default=2,help='number of epochs for train')
    parse.add_argument('--batch_size', type=int, default=100,help='number of epochs for train')
    parse.add_argument('--lr', type=float, default=0.0001,help='initial learning rate')
    parse.add_argument('--grad_clip', type=float, default=5.0,help='Gradient max clip!')
    parse.add_argument('--teacher_forcing_ratio',type=float,default=0.5,help='Teacher forcing ratio!')
    #
    parse.add_argument('--train_data_fileName',type=str,default='data/train_cp',help='Training raw data!')
    parse.add_argument('--test_data_fileName', type=str, default='data/test_cp',help='Testing raw data!')
    parse.add_argument('--save_path',type=str,default='save/',help='save path!')
    #
    parse.add_argument('--mode',type=str,default='train',help='Type of mode!')
    #
    return parse.parse_args()
#
args=parse_arguments()
#load data!
train_pairs, source_vocab_size, target_vocab_size = sent2id(args.train_data_fileName, reverse=False, mode='train')
test_pairs = sent2id(args.test_data_fileName, reverse=False, mode='test')
# print("source_vocab_size",source_vocab_size)
# print("target_vocab_size",target_vocab_size)
train_data_iter = batch_yeild(sentPairs=train_pairs,batch_size=args.batch_size)
test_data_iter = batch_yeild(sentPairs=test_pairs,batch_size=args.batch_size)
#初始化config！
config=Config(encoder_vocab_size=source_vocab_size,
              encoder_hidden_dim=args.encoder_hidden_dim,
              encoder_embed_dim=args.encoder_embed_dim,encoder_num_layers=args.encoder_num_layers,
              decoder_vocab_size=target_vocab_size,decoder_hidden_dim=args.decoder_hidden_dim,
              decoder_embed_dim=args.decoder_embed_dim,decoder_num_layers=args.decoder_num_layers,
              decoder_droupOut_p=args.decoder_droupOut_p,decoder_method=args.decoder_method,
              epoches=args.epochs,batch_size=args.batch_size,lr=args.lr,gradient_clip=args.grad_clip,
              teacher_forcing_ratio=args.teacher_forcing_ratio)
#
print("Models initializing....")

encoder=Encoder_for_batch(config)
decoder=Decoder_Luong_AM_for_batch(config)
if(config.use_cuda):
    seq2seq=Seq2seq_all_for_batch(encoder=encoder,decoder=decoder,Config=config).cuda()
else:
    seq2seq = Seq2seq_all_for_batch(encoder=encoder, decoder=decoder, Config=config)
print("Model structures:")
print(seq2seq)

optimizer=torch.optim.Adam(params=seq2seq.parameters(),lr=config.lr)

print("model parameters:")
# for num,par in enumerate(seq2seq.parameters()):
#     print("num:%d->"%(num),par)
#
if(args.mode=='train'):
    print("Training....")
    best_val_loss = None
    for epoch in range(config.epoches):
        train_one_epoch(epoch_num=epoch,seq2seq=seq2seq,
                        train_batch_iter=train_data_iter,
                        optimizer=optimizer,
                        clip=config.gradient_clip)
        #每一次epoch结束，在test集进行一次validation！
        val_loss=validation_after_one_epoch(seq2seq=seq2seq,validation_batch_iter=test_data_iter)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"% (epoch, val_loss, math.exp(val_loss)))
        #save_model:保存最优测试结果的epoch的Model！
        if not best_val_loss or val_loss < best_val_loss:
            print("Saving model...")
            torch.save(seq2seq.state_dict(), './save/seq2seq_%d.pt' % (epoch))
            best_val_loss = val_loss
    print("Training end....")
elif(args.mode=='inference'):
    print("Inference....")



