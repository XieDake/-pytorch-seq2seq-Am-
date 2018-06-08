# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
            load model and predict!
            加载保存的模型，并完成预测！
===============================================================
"""
import argparse,os,pickle
import torch
from config import Config
from models import Encoder_for_batch,Decoder_Luong_AM_for_batch
from data_helper import sent2id_for_inference
from train import predict_for_inference
PAD_token=0
SOS_token=1
EOS_token=2
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
    parse.add_argument('--decoder_method',type=str,default='general',help='Align methods of AM in decoder!')
    #
    parse.add_argument('--epochs', type=int, default=10,help='number of epochs for train')
    parse.add_argument('--batch_size', type=int, default=20,help='number of epochs for train')
    parse.add_argument('--lr', type=float, default=0.0001,help='initial learning rate')
    parse.add_argument('--grad_clip', type=float, default=5.0,help='Gradient max clip!')
    parse.add_argument('--teacher_forcing_ratio',type=float,default=0.5,help='Teacher forcing ratio!')
    #
    parse.add_argument('--Base_path', type=str, default='/Users/xieqiqi/Documents/pyenv/xuexi/pytorch-seq2seq-Am/pytorch_seq2seq_final/data/', help='Base path!')
    parse.add_argument('--Save_path', type=str, default='/Users/xieqiqi/Documents/pyenv/xuexi/pytorch-seq2seq-Am/pytorch_seq2seq_final/data/save_0/',
                       help='Save path!')
    #
    parse.add_argument('--mode',type=str,default='inference',help='Type of mode!')
    #
    return parse.parse_args()
print("==================================================================")
args=parse_arguments()
encoder_model_save_name=os.path.join(args.Save_path,"encoder_%d.pt")
decoder_model_save_name=os.path.join(args.Save_path,"decoder_%d.pt")
print("=============================定义网络===============================")
print("初始化config！")
config=Config(encoder_vocab_size=None,
              encoder_hidden_dim=args.encoder_hidden_dim,
              encoder_embed_dim=args.encoder_embed_dim,encoder_num_layers=args.encoder_num_layers,
              decoder_vocab_size=None,decoder_hidden_dim=args.decoder_hidden_dim,
              decoder_embed_dim=args.decoder_embed_dim,decoder_num_layers=args.decoder_num_layers,
              decoder_droupOut_p=args.decoder_droupOut_p,decoder_method=args.decoder_method,
              epoches=args.epochs,batch_size=args.batch_size,lr=args.lr,gradient_clip=args.grad_clip,
              teacher_forcing_ratio=args.teacher_forcing_ratio)
print("==================================================================")
print("Models initializing....")
encoder=Encoder_for_batch(config).cuda() if torch.cuda.is_available() else Encoder_for_batch(config)
decoder=Decoder_Luong_AM_for_batch(config).cuda() if torch.cuda.is_available() else Decoder_Luong_AM_for_batch(config)
# 定义网络
print("==========================加载网络参数==============================")
# 加载网络参数
encoder.load_state_dict(torch.load(encoder_model_save_name))
decoder.load_state_dict(torch.load(decoder_model_save_name))
print("======================预测========================================")
# 用新加载的参数进行预测
if(args.mode=='inference'):
    print("Inferencing...!")
    print("loading target id to wd!")
    with open("data/save/target_id2Wd", "rb") as fr:
        target_index2Wd = pickle.load(fr)
    while(1):
        print('Please input your sentence:')
        demo_sent = input()
        if demo_sent == '' or demo_sent.isspace():
            print('See you next time!')
            break
        else:
            demo_sent = sent2id_for_inference(input_sent=demo_sent)
            #predit!
            predict_ids=predict_for_inference(sent_id=demo_sent,encoder=encoder,decoder=decoder)
            #解码！
            translate=""
            for id in predict_ids:
                if(id==EOS_token):
                    print("source sentence:",demo_sent)
                    print("translate result:",translate)
                else:
                    translate+=(target_index2Wd[id]+' ')