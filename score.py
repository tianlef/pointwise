# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:38:45 2018

@author: tianle
"""
import pickle
import time
import numpy as np
import tensorflow as tf
class score:
    def __init__(self,sequence_length,batch_size,vocab_size,embedding_size,hidden_size,l2_reg_lambda=0.0,paras=None,learning_rate=1e-2,embeddings=None,loss="pair",trainable=True):
        self.learning_rate=learning_rate
        self.paras=paras
        self.l2_reg_lambda=l2_reg_lambda
        self.embeddings=embeddings
        self.sequence_length=sequence_length
        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.model_type="base"
        self.hidden_size=hidden_size
        self.is_train=trainable
       
        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [None,sequence_length], name="input_x_2")
        self.label=tf.placeholder(tf.float32, [None], name="label")
        
        
        
        self.updated_paras=[]
        # Embedding layer
        with tf.name_scope("embedding"):
            if self.paras==None:
                if self.embeddings ==None:
                    print ("random embedding")
                    #w是一个全部词行，和嵌入行的矩阵,
                    self.Embedding_W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="random_W") 
                else:
                    self.Embedding_W = tf.Variable(np.array(self.embeddings),name="embedding_W" ,dtype="float32",trainable=False)
            else:
                print ("load embeddings")
                self.Embedding_W=tf.Variable(self.paras[0],trainable=False,name="embedding_W")
            self.updated_paras.append(self.Embedding_W)

        self.l2_loss = tf.constant(0.0)
        #给矩阵算一个平均值
       
        for para in self.updated_paras:
            self.l2_loss+= tf.nn.l2_loss(para)

        with tf.name_scope("prepare"):
            q  =tf.nn.embedding_lookup(self.Embedding_W, self.input_x_1)
            pos=(self.Embedding_W, self.input_x_2)
            
        self.x12=self.data_concat(q,pos)

        fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size,forget_bias=1.0)
        bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size,forget_bias=1.0)
        with tf.variable_scope("dis"):
           
            self.outputs122,_,_=tf.contrib.rnn.static_bidirectional_rnn(fw_cell,bw_cell,self.x12,dtype=tf.float32,scope='xx')
            outputs12=tf.transpose(self.outputs122,[1,0,2])
            outputs12=tf.reshape(outputs12,[-1,2*hidden_size*sequence_length])        


        with tf.variable_scope('MLP'):
            self.W1 = tf.get_variable('w1',[2*hidden_size*sequence_length,1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.b1 = tf.get_variable('b1',[1],initializer=tf.constant_initializer(0.0))
            #self.W2 = tf.get_variable('w2',[self.hidden_size,1],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            #self.score=tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(outputs12, self.W1, self.b1)), self.W2)
            self.score=tf.nn.xw_plus_b(outputs12, self.W1, self.b1)
            self.score=tf.squeeze(self.score)

            tf.summary.histogram('score', self.score)


    def data_concat(self,q,a):
        
        mulq=tf.multiply(q,a)
        subq=tf.abs(tf.subtract(q,a))
        x=tf.concat([mulq,subq],2)
        x=tf.transpose(x,[1,0,2])
        x=tf.reshape(x,[-1,3*self.embedding_size])
        x=tf.split(x,self.sequence_length)
        return x
    def qa_concat(self,q,a):

    