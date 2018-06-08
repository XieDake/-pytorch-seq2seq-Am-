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
from models import Encoder_for_batch,Decoder_Luong_AM_for_batch
from train import train_one_epoch_0601,validation_after_one_epoch_0601

def parse_arguments():
    parse = argparse.ArgumentParser(description='Hyperparams of this project!')
    #
    parse.add_argument('--encoder_hidden_dim', type=int, default=256,help='Hidden dim of encoder!')
    parse.add_argument('--encoder_embed_dim', type=int, default=256,help='Embed dim of encoder!')
    parse.add_argument('--encoder_num_layers', type=int, default=1,help='Num layers of encoder!')
    #
    parse.add_argument('--decoder_hidden_dim', type=int, default=256,help='Hidden dim of decoder!')
    parse.add_argument('--decoder_embed_dim', type=int, default=256,help='Embed dim of decoder!')
    parse.add_argument('--decoder_num_layers', type=int, default=1,help='Num layers of decoder!')
    parse.add_argument('--decoder_droupOut_p',type=float,default=0.5,help='DroupOut prob of decoder!')
    parse.add_argument('--decoder_method',type=str,default='concat',help='Align methods of AM in decoder!')
    #
    parse.add_argument('--epochs', type=int, default=2,help='number of epochs for train')
    parse.add_argument('--batch_size', type=int, default=64,help='number of epochs for train')
    parse.add_argument('--lr', type=float, default=0.001,help='initial learning rate')
    parse.add_argument('--grad_clip', type=float, default=5.0,help='Gradient max clip!')
    parse.add_argument('--teacher_forcing_ratio',type=float,default=0.5,help='Teacher forcing ratio!')
    #
    parse.add_argument('--Base_path', type=str, default='/Users/xieqiqi/Documents/pyenv/xuexi/pytorch-seq2seq-Am/pytorch_seq2seq_final/data/', help='Base path!')
    parse.add_argument('--Save_path', type=str, default='/Users/xieqiqi/Documents/pyenv/xuexi/pytorch-seq2seq-Am/pytorch_seq2seq_final/data/save_0/',
                       help='Save path!')
    #
    parse.add_argument('--mode',type=str,default='train',help='Type of mode!')
    #
    return parse.parse_args()
#
args=parse_arguments()
print("==================Base Information:============================")
print("mode:",args.mode)
print("batch_size:",args.batch_size)
print("lr",args.lr)
print("grad_clip",args.grad_clip)
print("teacher_forcing_ratio",args.teacher_forcing_ratio)
print("===============================================================")
print("Path setting...")
train_data_fileName=os.path.join(args.Base_path,'train')
test_data_fileName=os.path.join(args.Base_path,'test')
save_path=args.Save_path
if(not os.path.exists(save_path)):
    os.mkdir(save_path)
#load data!
print("===============================================================")
print("Loading data and Data processing!")
train_pairs, source_vocab_size, target_vocab_size = sent2id(train_data_fileName, reverse=False, mode='train')
test_pairs = sent2id(test_data_fileName, reverse=False, mode='test')
train_sentence_pairs_size=len(train_pairs)
test_sentence_pairs_size=len(test_pairs)
print("train sentence pairs:",train_sentence_pairs_size)
print("test sentence pairs:",test_sentence_pairs_size)
print("===============================================================")
print("初始化config！")
config=Config(encoder_vocab_size=source_vocab_size,
              encoder_hidden_dim=args.encoder_hidden_dim,
              encoder_embed_dim=args.encoder_embed_dim,encoder_num_layers=args.encoder_num_layers,
              decoder_vocab_size=target_vocab_size,decoder_hidden_dim=args.decoder_hidden_dim,
              decoder_embed_dim=args.decoder_embed_dim,decoder_num_layers=args.decoder_num_layers,
              decoder_droupOut_p=args.decoder_droupOut_p,decoder_method=args.decoder_method,
              epoches=args.epochs,batch_size=args.batch_size,lr=args.lr,gradient_clip=args.grad_clip,
              teacher_forcing_ratio=args.teacher_forcing_ratio)
print("===============================================================")
print("Models initializing....")
encoder=Encoder_for_batch(config).cuda() if torch.cuda.is_available() else Encoder_for_batch(config)
decoder=Decoder_Luong_AM_for_batch(config).cuda() if torch.cuda.is_available() else Decoder_Luong_AM_for_batch(config)
optimizer_en=torch.optim.Adam(params=encoder.parameters(),lr=config.lr)
optimizer_de=torch.optim.Adam(params=decoder.parameters(),lr=config.lr)
print("===============================================================")
print("Model structures:")
print("Encoder:",encoder)
print("Decoder:",decoder)
print("===============================================================")
if(args.mode=='train'):
    print("Training....")
    best_val_loss = None
    #TODO:Early stop 机制，还木有加上！
    for epoch in range(config.epoches):
        #注意：因为使用的是generator所以，最好每一个epoch都重新产生一次！
        train_data_iter = batch_yeild(sentPairs=train_pairs, batch_size=args.batch_size)
        test_data_iter = batch_yeild(sentPairs=test_pairs, batch_size=args.batch_size)
        #
        train_one_epoch_0601(epoch_num=epoch,train_batch_iter=train_data_iter,
                             optimizer_en=optimizer_en,optimizer_de=optimizer_de,clip=config.gradient_clip,
                             teacher_forcing_ratio=config.teacher_forcing_ratio,encoder=encoder,decoder=decoder)
        #每一次epoch结束，在test集进行一次validation！
        val_loss=validation_after_one_epoch_0601(validation_batch_iter=test_data_iter,encoder=encoder,decoder=decoder,
                                                 data_size=test_sentence_pairs_size)
        print("Training on Epoch:%d finished! Val_loss=:%f | Val_pp:%fS"% (epoch, val_loss, math.exp(val_loss)))
        #save_model:保存最优测试结果的epoch的Model！
        if not best_val_loss or val_loss < best_val_loss:
            print("Time for saving model...")
            torch.save(encoder.state_dict(),os.path.join(save_path,'encoder_%d.pt' % (epoch)))
            torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder_%d.pt' % (epoch)))
            best_val_loss = val_loss
    print("Training end....")
