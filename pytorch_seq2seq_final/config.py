# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
===============================================================
"""
import torch
class Config:
    def __init__(self,encoder_vocab_size,encoder_hidden_dim,encoder_embed_dim,encoder_num_layers,
                 decoder_vocab_size,decoder_hidden_dim,decoder_embed_dim,decoder_num_layers,decoder_droupOut_p,decoder_method,
                 epoches,batch_size,lr,gradient_clip,teacher_forcing_ratio):
        self.use_cuda=torch.cuda.is_available()
        #Encoder
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_layers = encoder_num_layers
        #Decoder
        self.decoder_vocab_size = decoder_vocab_size
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_layers = decoder_num_layers
        self.decoder_droupOut_p = decoder_droupOut_p
        self.decoder_method = decoder_method
        #Train
        self.epoches = epoches
        self.batch_size = batch_size
        self.lr=lr
        self.gradient_clip = gradient_clip
        self.teacher_forcing_ratio = teacher_forcing_ratio