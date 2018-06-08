# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction: Training
===============================================================
"""
import torch.nn.functional as F
from torch.autograd import  Variable
from torch.nn.utils import clip_grad_norm
import math,random
import torch
import numpy as np

PAD_token=0
SOS_token=1
EOS_token=2
def train_one_epoch(epoch_num,seq2seq,train_batch_iter,optimizer,clip):
    '''
    完成一个batch数据的训练！
    :param seq2seq:组合模型
    :param batch_iter:batch_yeild
    :param lr:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :return:
    '''
    #TODO:训练模式：需要bp！
    seq2seq.train()
    target_vocab_size=seq2seq.decoder.vocab_size
    total_loss=0
    for step,batch_data in enumerate(train_batch_iter):
        #source_batch_pad, source_lengths, target_batch_pad, target_lengths
        source_batch_pad = batch_data[0]  #[B,max_seq_len]
        target_batch_pad = batch_data[2]  #[B,max_seq_len]
        #
        # print("pos1:",source_batch_pad.data.type)
        # print("pos2:",target_batch_pad.data.type)
        #
        optimizer.zero_grad()
        outPuts=seq2seq(source_batch_pad,target_batch_pad)#[seqLen,B,target_vocab_size]
        #
        # print("size of outPuts:",outPuts.data.size())
        #outPuts取timeStep从1往后的，flat展平[???,target_vocab_size]！
        #target_batch_pad从所有Batch的seqLen=1开始将所有的flat成一列！
        # loss = F.nll_loss(input=outPuts[1:].contiguous().view(-1,target_vocab_size),
        #                   target=target_batch_pad[:,1:].contiguous().view(-1),ignore_index=EOS_token)

        #计算loss的时候，target不能是Torch.cuda.LongTensor类型！
        if(torch.cuda.is_available()):
            target_batch_pad = batch_data[2].type(torch.LongTensor)

        loss = F.cross_entropy(input=outPuts[1:].contiguous().view(-1,target_vocab_size),
                          target=target_batch_pad[:,1:].contiguous().view(-1),ignore_index=EOS_token)
        #backwards
        loss.backward()
        #gradient_clip
        clip_grad_norm(seq2seq.parameters(),clip)
        #参数个更新
        optimizer.step()
        #
        total_loss+=loss.data[0]
        #
        print("loss of this step:{}".format(loss.data[0]))
        #每50步print损失loss信息！
        if(step%20==0 and step!=0):
            total_loss = total_loss / 100
            print("[epoch:%d][step:%d][loss:%5.2f][pp:%5.2f]" %(epoch_num,step,total_loss,math.exp(total_loss)))
            total_loss = 0
def validation_after_one_epoch(seq2seq,validation_batch_iter):
    '''
    每一次Epoch结束进行一次Validation！
    Scheduled Sampling只能在train的时候使用！eval的时候不用 Scheduled Sampling和Teacher Forcing！
    '''
    #TODO:Evaluation模式：不需要bp！
    seq2seq.eval()
    target_vocab_size=seq2seq.decoder.vocab_size
    total_loss=0
    data_size=0.0
    for step,batch_data in enumerate(validation_batch_iter):
        data_size+=len(batch_data)
        #source_batch_pad, source_lengths, target_batch_pad, target_lengths
        source_batch_pad=batch_data[0]#[B,max_seq_len]
        target_batch_pad=batch_data[2]#[B,max_seq_len]
        #
        outPuts=seq2seq(source_batch_pad,target_batch_pad)#[seqLen,B,target_vocab_size]
        #outPuts取timeStep从1往后的，flat展平[???,target_vocab_size]！
        #target_batch_pad从所有Batch的seqLen=1开始将所有的flat成一列！
        loss = F.nll_loss(input=outPuts[1:].contiguous().view(-1,target_vocab_size),
                          target=target_batch_pad[:,1:].contiguous().view(-1),
                          ignore_index=EOS_token)
        total_loss += loss.data[0]
    #
    return total_loss/data_size

def train_one_epoch_0601(epoch_num,train_batch_iter,optimizer_en,optimizer_de,clip,teacher_forcing_ratio,encoder,decoder):
    '''
    2018_06_01_改！
    '''
    # TODO:训练模式：需要bp！功能还需研究！
    encoder.train()
    decoder.train()
    #
    use_cuda=torch.cuda.is_available()
    total_loss = 0
    for step, batch_data in enumerate(train_batch_iter):
        # source_batch_pad, source_lengths, target_batch_pad, target_lengths,count
        source_batch_pad = batch_data[0]  # [B,max_seq_len]
        target_batch_pad = batch_data[2]  # [B,max_seq_len]
        #
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
        #
        bsz=source_batch_pad.size(0)
        #确定decoder需要的最大时间step！
        target_max_len=target_batch_pad.size(1)
        #target_vocab_size
        target_vocab_size=decoder.vocab_size
        #确定输出的维度！[seqLen,B,target_vocab_size]
        if(torch.cuda.is_available()):
            outPuts = Variable(torch.zeros(target_max_len, bsz, target_vocab_size)).cuda()
        else:
            outPuts = Variable(torch.zeros(target_max_len, bsz, target_vocab_size))
        #encoder
        #hidden:?*B*H——>(num_layers * num_directions, batch, hidden_size)
        #encoder_outPuts:T*B*H——>(seq_len, batch, hidden_size * num_directions)
        encoder_outPuts,hidden=encoder(source_batch_pad)#hidden=None
        #decoder:注意Batch形式每一次input的维度[B]
        #初始化decoder的hidden和第一次input！
        hidden=hidden[:encoder.num_layers]
        outPut=target_batch_pad[:,0]#SOS作为decoder的第一个输入！
        for len in range(1,target_max_len):
            #outPut:B*vocab_size
            outPut,hidden,_=decoder(outPut,hidden,encoder_outPuts)
            outPuts[len]=outPut
            #Prepare for next time step!
            is_teacher= random.random() > teacher_forcing_ratio
            top1=outPut.data.max(1)[1]#取index！
            if(use_cuda):
                outPut = Variable(target_batch_pad[:, len].data if is_teacher else top1).cuda()
            else:
                outPut = Variable(target_batch_pad[:, len].data if is_teacher else top1)
        # [seqLen,B,target_vocab_size]
        loss = F.cross_entropy(input=outPuts[1:].contiguous().view(-1, target_vocab_size),
                               target=target_batch_pad[:, 1:].contiguous().view(-1), ignore_index=EOS_token)
        # backwards
        loss.backward()
        # gradient_clip
        clip_grad_norm(decoder.parameters(), clip)
        clip_grad_norm(encoder.parameters(), clip)
        # 参数个更新
        optimizer_en.step()
        optimizer_de.step()
        #
        total_loss += loss.data[0]
        #
        # print("loss of this step:{}".format(loss.data[0]))
        # 每50步print损失loss信息！
        if (step % 20== 0 and step != 0):
            total_loss = total_loss / 20
            print("[epoch:%d][step:%d][loss:%f][pp:%f]" % (epoch_num, step, total_loss, math.exp(total_loss)))
            total_loss = 0

def validation_after_one_epoch_0601(validation_batch_iter,encoder,decoder,data_size):
    '''
    每一次Epoch结束进行一次Validation！
    Scheduled Sampling只能在train的时候使用！eval的时候不用 Scheduled Sampling和TeacherForcing！
    '''
    #TODO:Evaluation模式：不需要bp！功能还需研究！
    encoder.eval()
    decoder.eval()
    use_cuda=torch.cuda.is_available()
    total_loss=0
    for step, batch_data in enumerate(validation_batch_iter):
        # source_batch_pad, source_lengths, target_batch_pad, target_lengths
        source_batch_pad = batch_data[0]  # [B,max_seq_len]
        target_batch_pad = batch_data[2]  # [B,max_seq_len]
        #
        bsz=source_batch_pad.size(0)
        #确定decoder需要的最大时间step！
        target_max_len=target_batch_pad.size(1)
        #target_vocab_size
        target_vocab_size=decoder.vocab_size
        #确定输出的维度！[seqLen,B,target_vocab_size]
        if(torch.cuda.is_available()):
            outPuts = Variable(torch.zeros(target_max_len, bsz, target_vocab_size)).cuda()
        else:
            outPuts = Variable(torch.zeros(target_max_len, bsz, target_vocab_size))
        #encoder
        #hidden:?*B*H——>(num_layers * num_directions, batch, hidden_size)
        #encoder_outPuts:T*B*H——>(seq_len, batch, hidden_size * num_directions)
        encoder_outPuts,hidden=encoder(source_batch_pad)#hidden=None
        #decoder:注意Batch形式每一次input的维度[B]
        #初始化decoder的hidden和第一次input！
        hidden=hidden[:encoder.num_layers]
        outPut=target_batch_pad[:,0]#SOS作为decoder的第一个输入！
        for l in range(1,target_max_len):
            #outPut:B*vocab_size
            outPut,hidden,_=decoder(outPut,hidden,encoder_outPuts)
            outPuts[l]=outPut
            #Prepare for next time step!
            top1=outPut.data.max(1)[1]#取index！
            if(use_cuda):
                outPut = Variable(top1).cuda()
            else:
                outPut = Variable(top1)
        #计算loss！
        loss = F.cross_entropy(input=outPuts[1:].contiguous().view(-1, target_vocab_size),
                               target=target_batch_pad[:, 1:].contiguous().view(-1), ignore_index=EOS_token)
        #
        total_loss += loss.data[0]
    #
    return total_loss/data_size
#===============================================================
#===============================================================
#Prediction for inference!
def predict_for_inference(sent_id,encoder,decoder):
    '''
    进行一次预测，输出sentence！
    '''
    #TODO:Evaluation模式：不需要bp！功能还需研究！
    encoder.eval()
    decoder.eval()
    use_cuda=torch.cuda.is_available()
    #
    source_batch_pad = sent_id  # [B=1,seq_len]
    # encoder
    # hidden:?*B*H——>(num_layers * num_directions, batch, hidden_size)
    # encoder_outPuts:T*B*H——>(seq_len, batch, hidden_size * num_directions)
    encoder_outPuts, hidden = encoder(source_batch_pad)  # hidden=None
    # decoder:注意Batch形式每一次input的维度[B]
    # 初始化decoder的hidden和第一次input！
    hidden = hidden[:encoder.num_layers]
    outPuts=[]#保存decoder预测结果！
    outPut =[SOS_token]   # SOS作为decoder的第一个输入！
    #
    while(True):
        # outPut:B*vocab_size
        outPut, hidden, _ = decoder(outPut, hidden, encoder_outPuts)
        # Prepare for next time step!
        top1 = outPut.data.max(1)[1]  # 取index！
        outPuts.append(top1)
        if (use_cuda):
            outPut = Variable(top1).cuda()
        else:
            outPut = Variable(top1)
        #判断是否结束！
        if(top1==EOS_token):
            #eos结束标志位也保存！
            break
    #输出预测的wd的index！
    return outPuts