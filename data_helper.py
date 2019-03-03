# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:28:45 2018

@author: tianle
"""
import json
import numpy as np
import random
import math
import pickle
import csv

def build_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    vocab['<a>']=code
    code +=1
    filename="WebDoc.txt"
    f=open(filename,'r')
    line=f.readlines()
    for line in open(filename):
        items = line.strip().split(' ')
        for word in items:
            if not word in vocab:
                        vocab[word] = code
                        code += 1
    print("vocab done")
    return vocab
def load_vectors(vocab=None):
    if vocab==None:
        return
    vectors = {}
    for line in open('Webvectors.txt'):
        items = line.strip().split(' ')
        if (len(items) < 101):
            continue
        vec = []
        for i in range(1, 101):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    embeddings=[]
    for word in vocab:
        if word in vectors.keys():
            embeddings.append(vectors[word])
        elif word=='<a>':
            embeddings.append(np.zeros(100).tolist())
        else:
            random_embeding=np.random.uniform(-1,1,100).tolist()
            embeddings.append(random_embeding)
    return embeddings 

def read_alist(dataset="QA1"):
    alist = []
    for line in open(dataset+'/qe/term-train'):#what mean
        items = line.strip().split(' ')
        #items中含有标签 问题长度 问题编号 问题 回答 
        alist.append(items)  
        #print(items)
    print('read_alist done ......')
    return alist

def loadCandidateSamples(q,a,candidates,vocab):
    samples=[]
    
    for neg in candidates:
        samples.append((encode_sent(vocab,q,20),encode_sent(vocab,a,20),encode_sent(vocab,neg,20)))
    return  samples

#raw里加载了标签为1的例子里的所有内容
def read_raw(dataset="QA1"):
    raw = []
    for line in open(dataset+'/qe/groundTruth-train'):
        items = line.strip().split(' ')
        if items[0] == '1':
            raw.append(items)
    return raw
def encode_sent(vocab, string,size):
    x = []
    words = string.split('_')
    for i in range(size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x

def loadTestSet(dataset="QA1",filename="term"):
    testList = []
    for line in open(dataset+'/qe/'+filename):
        testList.append(line.strip())    # lower?
    return testList

#需要修改
def load_train_random(vocab, alist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for items in raw:
        items = raw[random.randint(0, len(raw) - 1)]
        nega = np.random.choice(alist)
        x_train_1.append(encode_sent(vocab, items[2], 20))
        x_train_2.append(encode_sent(vocab, items[3], 20))
        x_train_3.append(encode_sent(vocab, nega, 20))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)
#需要修改

def load_val_batch(testList, vocab, index, batch):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    #items中含有标签 问题长度 问题编号 问题 回答
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(testList)):
            true_index = len(testList) - 1
        items = testList[true_index].split(' ')
        x_train_1.append(encode_sent(vocab, items[3], 20))
        x_train_2.append(encode_sent(vocab, items[4], 20))
        x_train_3.append(int(items[1]))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)


#输出在候选例子中当前所选例子的目录
def batch_iter(data, batch_size, num_epochs=1, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch =int( math.ceil(len(data)/batch_size))#math.ceil 向上舍入
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min(((batch_num + 1) * batch_size), data_size)

            yield shuffled_data[start_index:end_index]

def qid(dataset="QA1"):
    with open(dataset, 'r', encoding='utf-8') as f:
        ss=f.readlines()
        number=[]
        for s in ss:
            item=s.strip().split(" ")
            num=len(item)-1
            number.append(num)
    return number
def qname(dataset="QA1"):
    qname=[]
    with open(dataset, 'r', encoding='utf-8') as f:
        ss=f.readlines()
        for s in ss:
            item=s.strip().split(" ")
            qa=str(item[0])
            qname.append(qa)
    return qname


def main():
    
    #vocab = build_vocab()
    #embeddings =load_vectors(vocab)
    #alist = read_alist(dataset="QA1")





    #raw = read_raw("QA1")
    number=qid("QA1")
    print(len(number))
    qs=qname("QA1")
    print(qs)
    print(len(qs))

    #test1List = loadTestSet("test1")
    #test2List= loadTestSet("test2")
    #devList= loadTestSet("dev")
    #testSet=[("test1",test1List),("test2",test2List),("dev",devList)]
'''
    name="vectors/pre-word2vec.txt"
    pickle.dump(embeddings, open(name, 'wb'))
    for _index ,pair in enumerate (raw):
        if _index %100==0:
            print( "have sampled %d pairs" % _index)
        q=pair[2]
        a=pair[3]

        pools=np.random.choice(alist,size=100)    
    
        canditates=loadCandidateSamples(q,a,pools,vocab)
        for batch in batch_iter(canditates,batch_size=50):
            print(batch)
'''


    
if __name__ == '__main__':
    main()
   
