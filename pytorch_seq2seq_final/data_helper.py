# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
            基于Seq2Seq实现en2ch翻译——DataHelper！
===============================================================
"""
import torch
import numpy as np
from torch.autograd import Variable
import pickle,random,re
################################################################
   ##################Tool of base class!################
################################################################
PAD_token=0
SOS_token=1
EOS_token=2
class Lang:
    def __init__(self,min_count):
        '''
        Initialize!
        '''
        self.min_count=min_count
        self.word2index={}
        self.wordCount={}
        self.index2word={0:"PAD",1:"SOS",2:"EOS"}
        self.n_word=3#wordNum指针!永远指向下一个word的pos！

    def seg_sentence(self,sentence):
        return self.filter(sentence).strip().split(' ')

    def words_index_words(self,sentence):
        words=self.seg_sentence(sentence)
        for wd in words:
            self.word_index_word(wd)

    def word_index_word(self,word):
        '''
        对word跟新：
            word2index
            index2word
            wordCount
        '''
        if(word not in self.word2index):
            #new wd！
            self.word2index[word]=self.n_word
            self.wordCount[word]=1
            self.index2word[self.n_word]=word
            self.n_word+=1
        else:
            #old wd！
            self.wordCount[word]+=1
    def filter(self,sent):
        '''
        大小写转小写，过滤掉非中文和非英文的东西！
        '''
        sent_process = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"," ", sent.lower())
        #
        return sent_process

    def trim(self):
        '''
        去掉低频词！
        '''
        keep_words=[]
        for wd in self.wordCount:
            if(self.wordCount[wd]>=self.min_count):
                keep_words.append(wd)

        print("原始词表size={},去掉低频词后词表size={},ratio={}".format(
            {len(keep_words)},{len(self.wordCount)},{(len(keep_words))/(len(self.wordCount))}))

        #重构Dict！
        self.word2index={}
        self.wordCount={}
        self.index2word={0:"PAD",1:"SOS",2:"EOS"}
        self.n_word=3#wordNum指针!永远指向下一个word的pos！

        for wd in keep_words:
            self.word_index_word(wd)
        #
        self.word2index["UNK"]=self.n_word
        self.n_word+=1
################################################################
################################################################
def loadFile(dataFileName,reverse=False):
    '''
    load data!
    返回sentence 句对！
    [[sourceSent1,targetSent1],[sourceSent2,targetSent2],[sourceSent3,targetSent3],...]
    默认：
        source=EN
        target=CH
    '''
    sentPairs=[]
    with open(dataFileName,'r',encoding='UTF-8')as r:
        lines=r.readlines()
        for index,line in enumerate(lines):
            lsp=line.strip().split('\t')
            if(len(lsp)!=2):
                print("load data error ar row:{}".format({index}))
                continue
            else:
                sent_pair=[]
                en_sent = lsp[0]
                ch_sent = lsp[1]
                if (not reverse):
                    sent_pair.append(en_sent)
                    sent_pair.append(ch_sent)
                    sentPairs.append(sent_pair)
                else:
                    sent_pair.append(ch_sent)
                    sent_pair.append(en_sent)
                    sentPairs.append(sent_pair)
    #
    return sentPairs

def sent2id(dataFileName,reverse=False,mode='train'):
    '''
    完成source与target，id话，返回pairs！Dict 保存！
    '''
    if mode=='train':
        print("loading data....")
        sentPairs = loadFile(dataFileName=dataFileName, reverse=reverse)
        # source process!
        print("Source data processing....")
        source_sents2id = []
        source_data_helper = Lang(min_count=1)
        for sentPair in sentPairs:
            source_data_helper.words_index_words(sentPair[0].lower())

        source_data_helper.trim()
        source_word2index = source_data_helper.word2index
        source_index2word = source_data_helper.index2word
        with open("data/save/source_W2Id", 'wb') as fw:
            pickle.dump(source_word2index, fw)
        with open("data/save/source_id2Wd", 'wb') as fw:
            pickle.dump(source_index2word, fw)
        # source每一句话最后一个统一添加EOS！
        for sentPair in sentPairs:
            s_sent2id = []
            s_sent_wds = sentPair[0].split(' ')
            for wd in s_sent_wds:
                if (wd not in source_word2index):
                    s_sent2id.append(source_word2index["UNK"])
                else:
                    s_sent2id.append(source_word2index[wd])
            #source每一句话最后一个统一添加EOS！
            s_sent2id.append(EOS_token)
            source_sents2id.append(s_sent2id)
        # target process!
        print("Target data processing....")
        target_sents2id = []
        target_data_helper = Lang(min_count=1)
        for sentPair in sentPairs:
            target_data_helper.words_index_words(sentPair[1].lower())

        target_data_helper.trim()
        target_word2index = target_data_helper.word2index
        target_index2word = target_data_helper.index2word
        with open("data/save/target_W2Id", 'wb') as fw:
            pickle.dump(target_word2index, fw)
        with open("data/save/target_id2Wd", 'wb') as fw:
            pickle.dump(target_index2word, fw)
        # target的开始第一个统一SOS，最后一个统一添加EOS！
        for sentPair in sentPairs:
            t_sent2id = []
            t_sent2id.append(SOS_token)
            t_sent_wds = sentPair[1].split(' ')
            for wd in t_sent_wds:
                if (wd not in target_word2index):
                    t_sent2id.append(target_word2index["UNK"])
                else:
                    t_sent2id.append(target_word2index[wd])
            t_sent2id.append(EOS_token)
            target_sents2id.append(t_sent2id)
        # combin
        sents2Id_pairs = zip(source_sents2id, target_sents2id)
        #训练返回：data，source_vocab_size，target_vocab_size
        return list(sents2Id_pairs),source_data_helper.n_word,target_data_helper.n_word
    else:
        #test!
        print("loading tesing data....")
        sentPairs = loadFile(dataFileName=dataFileName, reverse=reverse)
        # source process!
        print("Source data processing....")
        source_sents2id = []
        print("Loading source dict....")
        with open("data/save/source_W2Id", "rb") as fr:
            source_word2index = pickle.load(fr)
        # source每一句话最后一个统一添加EOS！
        for sentPair in sentPairs:
            s_sent2id = []
            s_sent_wds = sentPair[0].split(' ')
            for wd in s_sent_wds:
                if (wd not in source_word2index):
                    s_sent2id.append(source_word2index["UNK"])
                else:
                    s_sent2id.append(source_word2index[wd])
            # source每一句话最后一个统一添加EOS！
            s_sent2id.append(EOS_token)
            source_sents2id.append(s_sent2id)
        # target process!
        print("Target data processing....")
        target_sents2id = []
        print("Loading target dict....")
        with open("data/save/target_W2Id", "rb") as fr:
            target_word2index = pickle.load(fr)
        # target的开始第一个统一SOS，最后一个统一添加EOS！
        for sentPair in sentPairs:
            t_sent2id = []
            t_sent2id.append(SOS_token)
            t_sent_wds = sentPair[1].split(' ')
            for wd in t_sent_wds:
                if (wd not in target_word2index):
                    t_sent2id.append(target_word2index["UNK"])
                else:
                    t_sent2id.append(target_word2index[wd])
            t_sent2id.append(EOS_token)
            target_sents2id.append(t_sent2id)
        # combin
        sents2Id_pairs = zip(source_sents2id, target_sents2id)
        return list(sents2Id_pairs)

def pad_sent(seq,max_length):
    '''
    每一块Batch数据进行Pading！
    '''
    seq+=[PAD_token for i in range(max_length-len(seq))]
    return seq

def batch_yeild(sentPairs,batch_size):
    '''
    每一个batch记得要PAD一下！
    维度要求：B*seqLen
    '''
    random.shuffle(sentPairs)
    source_batch, target_batch = [], []
    source_lengths, target_lengths = [], []
    for source_sent, target_sent in sentPairs:
        if len(source_batch) == batch_size:
            seq_pairs=sorted(zip(source_batch,target_batch),key=lambda p:len(p[0]),reverse=True)
            s_batch, t_batch=zip(*seq_pairs)
            #
            for s in s_batch:
                source_lengths.append(len(s))
            source_batch_pad=[pad_sent(s,max(source_lengths)) for s in s_batch]
            #
            for t in t_batch:
                target_lengths.append(len(t))

            target_batch_pad = [pad_sent(t, max(target_lengths)) for t in t_batch]
            #
            if torch.cuda.is_available():
                source_batch_pad=Variable(torch.from_numpy(np.array(source_batch_pad))).cuda()
                target_batch_pad=Variable(torch.from_numpy(np.array(target_batch_pad))).cuda()
            else:
                source_batch_pad=Variable(torch.from_numpy(np.array(source_batch_pad)))
                target_batch_pad=Variable(torch.from_numpy(np.array(target_batch_pad)))
            yield source_batch_pad,source_lengths,target_batch_pad,target_lengths
            source_batch, target_batch = [], []
            source_lengths, target_lengths = [], []
        source_batch.append(source_sent)
        target_batch.append(target_sent)
    if len(source_batch) != 0:
        seq_pairs = sorted(zip(source_batch, target_batch), key=lambda p: len(p[0]), reverse=True)
        s_batch, t_batch = zip(*seq_pairs)
        #
        for s in s_batch:
            source_lengths.append(len(s))
        source_batch_pad = [pad_sent(s, max(source_lengths)) for s in source_batch]
        #
        for t in t_batch:
            target_lengths.append(len(t))
        target_batch_pad = [pad_sent(t, max(target_lengths)) for t in target_batch]
        #Variable——cuda！
        if torch.cuda.is_available():
            source_batch_pad = Variable(torch.from_numpy(np.array(source_batch_pad))).cuda()
            target_batch_pad = Variable(torch.from_numpy(np.array(target_batch_pad))).cuda()
        else:
            source_batch_pad = Variable(torch.from_numpy(np.array(source_batch_pad)))
            target_batch_pad = Variable(torch.from_numpy(np.array(target_batch_pad)))
        yield source_batch_pad, source_lengths, target_batch_pad, target_lengths

#data_helper for inference!
def filter(sent):
    '''
    大小写转小写，过滤掉非中文和非英文的东西！
    '''
    sent_process = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"," ", sent.lower()).strip().split(' ')
    #
    return sent_process

def sent2id_for_inference(input_sent):
    '''
    Inference input sentence process！
    '''
    #source process!
    print("Source data processing....")
    s_sent2id = []
    print("Loading source dict....")
    with open("data/save/source_W2Id", "rb") as fr:
        source_word2index = pickle.load(fr)
    #source每一句话最后一个统一添加EOS！
    for wd in filter(sent=input_sent):
        if (wd not in source_word2index):
            s_sent2id.append(source_word2index["UNK"])
        else:
            s_sent2id.append(source_word2index[wd])
    #source每一句话最后一个统一添加EOS！
    s_sent2id.append(EOS_token)
    if(torch.cuda.is_available()):
        out = Variable(torch.from_numpy(np.array(s_sent2id))).cuda().unsqueeze(0)
    else:
        out = Variable(torch.from_numpy(np.array(s_sent2id))).unsqueeze(0)
    #
    return out
