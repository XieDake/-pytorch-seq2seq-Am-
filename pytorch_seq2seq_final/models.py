# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
              Seq2seq for Nnm!
              Not batch!
===============================================================
"""
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
#=================================================================
#=============================Encoders============================
#=================================================================
class Encoder_no_batch(torch.nn.Module):
    '''
    not for batch!
    '''
    def __init__(self,Config):
        super(Encoder_no_batch,self).__init__()
        self.use_cuda=Config.use_cuda
        #Parameters And Base Structers!
        self.vocab_size=Config.encoder_vocab_size
        self.hidden_dim=Config.encoder_hidden_dim
        self.embed_dim=Config.encoder_embed_dim
        self.num_layers=Config.encoder_num_layers

        self.embedings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        self.gru=torch.nn.GRU(self.embed_dim,self.hidden_dim)

    def forward(self,input,hidden=None):
        '''
        以单个wd_id作为基本输入单位，不是一个batch，也不是一整句话！
        :param input:
        :param hidden:
        :return:
        '''
        wd_embed=self.embedings(input).view(1,1,-1)#变成一个tensor(3D)：[1,1,embed_dim]
        #开始gru运算！
        output=wd_embed
        for layer in range(self.num_layers):
            output,hidden=self.gru(output,hidden)
        #ready for next time step！
        return output,hidden

    # def hidden_init(self):
    #     '''
    #     Encoder hidden initialization!
    #     :return:
    #     '''
    #     if self.use_cuda:
    #         return torch.zeros(1,1,self.hidden_dim).cuda()
    #     else:
    #         return torch.zeros(1,1,self.hidden_dim)

class Encoder_for_batch(torch.nn.Module):
    '''
    Support for batch!
    '''
    def __init__(self,Config):
        super(Encoder_for_batch,self).__init__()
        self.use_cuda=Config.use_cuda
        #Parameters And Base Structers!
        self.vocab_size=Config.encoder_vocab_size
        self.hidden_dim=Config.encoder_hidden_dim
        self.embed_dim=Config.encoder_embed_dim
        self.num_layers=Config.encoder_num_layers

        self.embedings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        self.gru=torch.nn.GRU(self.embed_dim,self.hidden_dim)

    def forward(self,input_batch,hidden=None):
        '''
        因为不需要得到每一个时间step的gru计算的结果，所以input长度为max即可！
        :param input:B*T
        :param hidden:
        :return:
        '''
        wd_embed=self.embedings(input_batch).transpose(0, 1)#B*T*embed_dim->T*B*embed_dim
        #开始gru运算！
        output=wd_embed
        for layer in range(self.num_layers):
            output,hidden=self.gru(output,hidden)
        #hidden:?*B*H——>(num_layers * num_directions, batch, hidden_size)
        #outPut:T*B*H——>(seq_len, batch, hidden_size * num_directions)
        return output,hidden
#=================================================================
#=============================Attention Mechanism=================
#=================================================================

class Attention_Bahdanau_for_batch(torch.nn.Module):
    '''
    Suport for Batch!
    实现Bahdanau的AM！
    根据hidden和encoder_outPuts,输出attention的weight：aij（softmax的结果！）
    '''
    def __init__(self,hidden_dim):
        super(Attention_Bahdanau_for_batch,self).__init__()
        # Parameters And Base Structers!
        self.hidden_dim=hidden_dim
        #AM_base_structures!
        # 论文中所说的使用前馈神经网络来计算score(i,j)!输入si-1，与hj，batch运算则是输入cat[si-1(expand T),encoder outputs(T)]
        #输入：B*T*2H->输出：B*T*H
        self.attn=torch.nn.Linear(2*self.hidden_dim,self.hidden_dim)
        #计算score还需要Va！使用nn.Parameter()将其变成Modul的属性，自动加入Parameters的迭代器中！
        #Va随机初始化，并且经过归一化！
        self.v=torch.nn.Parameter(torch.rand(self.hidden_dim))
        bound=1./(math.sqrt(self.v.size(0)))
        self.v.data.uniform_(-bound,bound)

    def forward(self,hidden,encoder_outPuts):
        '''
        :param hidden:1*B*H——>(num_layers * num_directions, batch, hidden_size)
        :param encoder_outPuts: T*B*H——>(seq_len, batch, hidden_size * num_directions)
        :return:
        '''
        T=encoder_outPuts.size(0)
        #上一时刻的hidden——>copyT份（T*B*H）
        h=hidden.repeat(T,1,1)#T*B*H
        #计算attention_weight:Va*atten()
        #B*T*2H->B*T*H
        #encoder_outPuts.t:B*T*H——h.t:B*T*H——在第二维cat（h.t,encoder_outPuts,dim=2）
        energy=torch.cat((h.transpose(0,1),encoder_outPuts.transpose(0,1)),dim=2)
        #batch运算，需要对Va——copy B份，即每一句话都对应一个权重Va！
        Va=self.v.repeat(energy.size(0),1).unsqueeze(1)#B*1*H
        #B*1*H_B*H*T=B*1*T
        score=torch.bmm(Va,energy.transpose(1,2))#B*1*T
        weights=F.softmax(score.squeeze(1))#B*T
        return weights

class Attention_Luong_no_batch(torch.nn.Module):
    '''
    Global Attenion Mechanism!
    输出：Attention weight
    '''
    def __init__(self,hidden_dim,method):
        super(Attention_Luong_no_batch,self).__init__()
        self.use_cuda=torch.cuda.is_available()
        # Parameters And Base Structers!
        self.hidden_dim=hidden_dim
        self.method=method
        # AM_base_structures!
        if(self.method=="general"):
            self.attn=torch.nn.Linear(self.hidden_dim,self.hidden_dim)
        elif(self.method=="concat"):
            self.attn=torch.nn.Linear(self.hidden_dim*2,self.hidden_dim)
            #Va
            self.v = torch.nn.Parameter(torch.rand(1,self.hidden_dim))
            bound = 1. / (math.sqrt(self.v.size(1)))
            self.v.data.uniform_(-bound, bound)

    def forward(self,hidden,encoder_outputs):
        '''
        :param hidden:[1,1,H]——当前时刻的隐层state
        :param encoder_outputs:[seqlen,1,H]
        :return:
        '''
        seq_len=encoder_outputs.size(0)
        scores=Variable(torch.zeros(seq_len))#seq_len
        scores=scores.cuda() if self.use_cuda else scores

        for i in range(seq_len):
            scores[i]=self.score(hidden,encoder_outputs[i])
        #[seqLen]->[1,1,segLen]
        weights=F.softmax(scores).unsqueeze(0).unsqueeze(0)#[1,1,segLen]
        return weights

    def score(self,hidden,encoder_output):
        '''
        :param hidden:
        :param encoder_outputs:
        :return:
        '''
        if(self.method=="dot"):
            #ht.hs
            return hidden.dot(encoder_output)
        elif(self.method=="general"):
            #ht.W.hs
            return hidden.dot(self.attn(encoder_output))
        elif(self.method=="concat"):
            #ht:[1,1,H],hs[1,1,H]
            #Va.W.cat[ht;hs]
            return self.v.unsqueeze(0).dot(self.attn(torch.cat((hidden,encoder_output),dim=2)))

class Attention_Luong_for_batch(torch.nn.Module):
    '''
    Global Attenion Mechanism!
    Support For Batch!
    输出：Attention weight!
    '''
    def __init__(self,hidden_dim,method):
        super(Attention_Luong_for_batch,self).__init__()
        self.use_cuda=torch.cuda.is_available()
        # Parameters And Base Structers!
        self.hidden_dim=hidden_dim
        self.method=method
        # AM_base_structures!
        if(self.method=="general"):
            self.attn=torch.nn.Linear(self.hidden_dim,self.hidden_dim)
        elif(self.method=="concat"):
            self.attn=torch.nn.Linear(self.hidden_dim*2,self.hidden_dim)
            #Va
            self.v = torch.nn.Parameter(torch.rand(1,self.hidden_dim))
            # self.v = torch.nn.Parameter(torch.FloatTensor(1,hidden_dim))
            bound = 1. / (math.sqrt(self.v.size(1)))
            self.v.data.uniform_(-bound, bound)

    def forward(self,hidden,encoder_outputs):
        '''
        :param hidden:[1,B,H]——当前时刻的隐层state
        :param encoder_outputs:[seqlen,B,H]
        :return:
        '''
        seq_lens=encoder_outputs.size(0)
        this_batch_size=encoder_outputs.size(1)
        scores=Variable(torch.zeros(this_batch_size,seq_lens))#this_batch_size*seq_len
        scores=scores.cuda() if self.use_cuda else scores
        for btz in range(this_batch_size):
            for len in range(seq_lens):
                scores[btz,len] = self.score(hidden[:,btz], encoder_outputs[len,btz].unsqueeze(0))
        #[seqLen]->[1,B,segLen]
        #weights=F.softmax(scores).unsqueeze(0)#[1,B,segLen]
        return F.softmax(scores).unsqueeze(0)

    def score(self,hidden,encoder_output):
        '''
        :param hidden:
        :param encoder_outputs:
        :return:
        '''
        if(self.method=="dot"):
            #ht.hs
            return hidden.dot(encoder_output)
        elif(self.method=="general"):
            #ht.W.hs
            return hidden.squeeze(0).dot(self.attn(encoder_output).squeeze(0))
        elif(self.method=="concat"):
            #ht:[1,H],hs[1,H]
            #Va.W.cat[ht;hs]
            tmp=self.attn(torch.cat((hidden,encoder_output),dim=1))
            #
            return self.v.squeeze(0).dot(tmp.squeeze(0))
#=================================================================
#=============================Decoders============================
#=================================================================
class Decoder_no_batch_no_Am(torch.nn.Module):
    '''
    not for batch
    Decoder without AM!
        1：修正理解上的一个错误！context vector 在Decoder中的使用！
        2：按道理来说Decoder的初始hidden是context vector——不用初始化咯！
    '''
    def __init__(self,Config):
        super(Decoder_no_batch_no_Am,self).__init__()
        self.use_cuda=Config.use_cuda
        #Paremeters and base structures!
        self.vocab_size=Config.decoder_vocab_size
        self.hidden_dim=Config.decoder_hidden_dim
        self.embed_dim=Config.decoder_embed_dim
        self.num_layers=Config.decoder_num_layers

        self.embedings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        self.gru=torch.nn.GRU(self.embed_dim,self.hidden_dim)
        self.out=torch.nn.Linear(self.hidden_dim,self.vocab_size)

    def forward(self,input,hidden):
        '''
        :param input:
        :param hidden:
        :return:
        '''
        #wd_embeding
        wd_embed=self.embedings(input).view(1,1,-1)#变成Tensor：3D！
        #input——>relu
        outPut=F.relu(wd_embed)
        #gru运算
        for layer in range(self.num_layers):
            outPut,hidden=self.gru(outPut,hidden)
        #输出——————————注意： 如果使用log_softmax,则要一定使用torch.nn.NLLLoss损失！如果使用softmax，则使用cross_entropy损失！
        outPut=F.log_softmax(self.out(outPut))
        #
        return outPut,hidden

class Decoder_Bahdanau_AM_for_batch(torch.nn.Module):
    '''
    Support for batch!
    '''
    def __init__(self,Config):
        super(Decoder_Bahdanau_AM_for_batch,self).__init__()
        self.use_cuda=Config.use_cuda
        #Paremeters and base structures!
        self.vocab_size=Config.decoder_vocab_size
        self.hidden_dim=Config.decoder_hidden_dim
        self.embed_dim=Config.decoder_embed_dim
        self.num_layers=Config.decoder_num_layers
        self.droupOut_p=Config.decoder_droupOut_p

        self.embedings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        self.droupOut=torch.nn.Dropout(self.droupOut_p)
        self.gru=torch.nn.GRU(self.embed_dim+self.hidden_dim,self.hidden_dim)
        self.out=torch.nn.Linear(self.hidden_dim*2,self.vocab_size)
        self.AM=Attention_Bahdanau_for_batch(hidden_dim=self.hidden_dim)

    def forward(self,input_batch,hidden_prev,encoder_outPuts):
        '''
        因为是Batch training，所以和一句话的forward不同，hidden——hidden_prev！
        :param input_batch(current input word or last output word)(id):[B]：1D一维的！
        :param hidden_prev:1*B*H——>(???)
        :param encoder_outPuts:T*B*H
        :return:
        '''
        #1：embedding!
        embed_batch=self.embedings(input_batch).unsqueeze(0)#B*H->1*B*H
        #2：embedding->droupOut!
        embed_batch = self.droupOut(embed_batch)
        #3：attention weights
        atten_weight=self.AM(hidden_prev,encoder_outPuts).unsqueeze(1)#B*T->B*1*T
        #4：ct:encoder_outPuts:T*B*H_atten_weight:B*1*T
        context_vector=torch.bmm(atten_weight,encoder_outPuts.transpose(0,1)).transpose(0,1)#B*1*H->1*B*H
        #5：计算St：gru计算！f(St-1,cat[yi-1,ci])
        #注意每一个batch只进行一次GRU计算，注意区分非batch的循环计算！
        rnn_input=torch.cat((embed_batch,context_vector),dim=2).transpose(0,1)#1*B*2H
        #rnn_input:1*B*2H
        #hidden:1*B*H——>(num_layers * num_directions, batch, hidden_size)
        #outPut:1*B*H——>(seq_len, batch, hidden_size * num_directions)
        outPut=rnn_input
        hidden=hidden_prev
        # for layer in range(self.num_layers-1):
        #     outPut,hidden=self.gru(outPut,hidden)
        outPut, hidden = self.gru(outPut, hidden)
        #6：计算预测y：y=out(cat[gru_out_t,ct])
        outPut=outPut.squeeze(0)#1*B*H->B*H
        context_vector=context_vector.squeeze(0)#1*B*H->B*H
        outs=self.out(torch.cat((outPut,context_vector),dim=1))#B*2H->B*vocab_size
        #7:logits—>B*vocab_size
        logits=F.log_softmax(outs)
        return logits,hidden,atten_weight

class Decoder_Luong_AM_no_batch(torch.nn.Module):
    '''
    基于Luong的G_AM的Decoder的实现！
    Not support for batch！
    '''
    def __init__(self,Config):
        super(Decoder_Luong_AM_no_batch,self).__init__()
        self.use_cuda=Config.use_cuda
        #Paremeters and base structures!
        self.vocab_size=Config.decoder_vocab_size
        self.hidden_dim=Config.decoder_hidden_dim
        self.embed_dim=Config.decoder_embed_dim
        self.num_layers=Config.decoder_num_layers
        self.droupOut_p=Config.decoder_droupOut_p
        self.method=Config.decoder_method

        self.embedings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        self.droupOut=torch.nn.Dropout(self.droupOut_p)
        self.gru_he=torch.nn.GRU(self.embed_dim+self.hidden_dim,self.hidden_dim)
        self.gru_me = torch.nn.GRU(self.embed_dim,self.hidden_dim)
        self.out=torch.nn.Linear(self.hidden_dim*2,self.vocab_size)
        self.AM=Attention_Luong_no_batch(hidden_dim=self.hidden_dim,method=self.method)

    def forward(self,input,hidden,encoder_outputs):
        '''
        Not support for batch！
        :param input(id):[1]_1D
        :param hidden:1*1*H
        :param encoder_outputs:seq_len*1*H
        :return:
        '''
        #1:embeding!
        wd_embed=self.embedings(input).view(1,1,-1)#1*1*H
        wd_embed=self.droupOut(wd_embed)
        #2:计算当前时刻隐层输出！论文中我的理解是：gru(yt-1,ht-1)！
        outPut=wd_embed
        outPut,hidden=self.gru_me(outPut,hidden)
        #但是看有人是这么实现的gru(cat(yt - 1, last_Context_Vector), ht - 1)
        # outPut=torch.cat((wd_embed,hidden),dim=2)#1*1*2H
        # outPut,hidden=self.gru_he(outPut,hidden)#[1,1,H],[1,1,H]
        #3:计算contex tVector
        #我根据论文理解：这里计算attn_weights使用的是当前时刻的隐层state，而不是当前时刻hidden输出！
        attn_weights=self.AM(hidden,encoder_outputs)#1*1*seqLen
        #但是有人是使用当前时刻gru输出来计算的！????我感觉我是对的！
        # attn_weights=self.AM(outPut,encoder_outputs)#1*1*segLen
        #context_vector[1,1,H]:attn_weights x encoder_outPuts.transpose(0,1)
        context_vector=torch.bmm(attn_weights,encoder_outputs.transpose(0,1))#1*1*H
        #4:计算yt——out！
        #根据gru的当前时刻的输出和context_vector,预测yt！
        out=self.out(torch.cat((context_vector,outPut),dim=2))#1*1*vocab_size
        #logits
        logits=F.log_softmax(out.squeeze(0))#1*vocab_size
        #
        return logits,hidden,context_vector,attn_weights

class Decoder_Luong_AM_for_batch(torch.nn.Module):
    '''
    基于Luong的G_AM的Decoder的实现！
    Support for batch！
    '''
    def __init__(self,Config):
        super(Decoder_Luong_AM_for_batch,self).__init__()
        self.use_cuda=Config.use_cuda
        #Paremeters and base structures!
        self.vocab_size=Config.decoder_vocab_size
        self.hidden_dim=Config.decoder_hidden_dim
        self.embed_dim=Config.decoder_embed_dim
        self.num_layers=Config.decoder_num_layers
        self.droupOut_p=Config.decoder_droupOut_p
        self.method=Config.decoder_method

        self.embedings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        self.droupOut=torch.nn.Dropout(self.droupOut_p)
        self.gru_he=torch.nn.GRU(self.embed_dim+self.hidden_dim,self.hidden_dim)
        self.gru_me = torch.nn.GRU(self.embed_dim,self.hidden_dim)
        self.out=torch.nn.Linear(self.hidden_dim*2,self.vocab_size)
        self.AM=Attention_Luong_for_batch(hidden_dim=self.hidden_dim,method=self.method)

    def forward(self,input_batch,hidden_prev,encoder_outputs):
        '''
        Not support for batch！
        :param input_batch(id):[B]：1D一维的！
        :param hidden_prev:1*B*H
        :param encoder_outputs:seq_len*B*H
        :return:
        '''
        #1：embedding!
        # if(self.use_cuda):
        #     embed_batch = self.embedings(input_batch).unsqueeze(0).cuda()  # B*H->1*B*H
        # else:
        #     embed_batch = self.embedings(input_batch).unsqueeze(0)  # B*H->1*B*H
        embed_batch = self.embedings(input_batch).unsqueeze(0)  # B*H->1*B*H
        #2：embedding->droupOut!
        embed_batch = self.droupOut(embed_batch)
        outPut=embed_batch
        outPut,hidden=self.gru_me(outPut,hidden_prev)#[1,B,H],[1,B,H]
        #3:计算context Vector
        attn_weights=self.AM(hidden,encoder_outputs)#1*B*seqLen
        #context_vector[B,1,H]:attn_weights.transpose(0,1) x encoder_outPuts.transpose(0,1)
        context_vector=torch.bmm(attn_weights.transpose(0,1),encoder_outputs.transpose(0,1))#B*1*H
        #4:计算yt——out！
        #根据gru的当前时刻的输出和context_vector,预测yt！
        out=self.out(torch.cat((context_vector,outPut.transpose(0,1)),dim=2))#B*1*2H->B*1*vocab_size
        #logits
        # logits=F.log_softmax(out.squeeze(1))#B*vocab_size
        out=out.squeeze(1)
        #
        return out,hidden,attn_weights
#=================================================================
#============================Seq2Seq Main Structure!==============
#=================================================================
class Seq2seq_all_for_batch(torch.nn.Module):
    '''
    Seq2Seq的main 入口！
    输入：source_batch：[B,seqLen]，target_batch:[B,seqLen]
    输出：[B,seqLen,target_vocab_size]该batch每一个的预测输出的logts矩阵！
    '''
    def __init__(self,encoder,decoder,Config):
        super(Seq2seq_all_for_batch,self).__init__()
        self.use_cuda = Config.use_cuda
        # Paremeters and base structures!
        self.encoder=encoder
        self.decoder=decoder
        self.teacher_forcing_ratio=Config.teacher_forcing_ratio

    def forward(self, source_batch,target_batch):
        '''
        :param source_batch:B*seqLen
        :param target_batch:B*seqLen
        '''
        #BatchSize
        bsz=source_batch.size(0)
        #确定decoder需要的最大时间step！
        target_max_len=target_batch.size(1)
        #target_vocab_size
        target_vocab_size=self.decoder.vocab_size
        #确定输出的维度！[seqLen,B,target_vocab_size]
        outPuts=Variable(torch.zeros(target_max_len,bsz,target_vocab_size))
        #encoder
        #hidden:?*B*H——>(num_layers * num_directions, batch, hidden_size)
        #encoder_outPuts:T*B*H——>(seq_len, batch, hidden_size * num_directions)
        encoder_outPuts,hidden=self.encoder(source_batch)#hidden=None
        #decoder:注意Batch形式每一次input的维度[B]
        #初始化decoder的hidden和第一次input！
        hidden=hidden[:self.encoder.num_layers]
        outPut=target_batch[:,0]#SOS作为decoder的第一个输入！
        for len in range(1,target_max_len):
            #outPut:B*vocab_size
            outPut,hidden,_=self.decoder(outPut,hidden,encoder_outPuts)
            outPuts[len]=outPut
            #Prepare for next time step!
            #TODO: Scheduled Sampling只能在train的时候使用！eval的时候不用 Scheduled Sampling和TeacherForcing！
            is_teacher= random.random() > self.teacher_forcing_ratio
            top1=outPut.data.max(1)[1]#取index！
            if(self.use_cuda):
                outPut = Variable(target_batch[:, len].data if is_teacher else top1).cuda()
            else:
                outPut = Variable(target_batch[:, len].data if is_teacher else top1)
        #outPuts:[seqLen,B,target_vocab_size]——其实是从1开始存的，0时间步结果为0！
        return outPuts