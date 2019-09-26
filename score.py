# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:38:45 2018

@author: tianle
"""
import pickle
import time
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.layers.fully_connected as linear
class score:
    def __init__(self,sequence_length,batch_size,vocab_size,embedding_size,hidden_size,l2_reg_lambda=0.0,paras=None,learning_rate=1e-2,embeddings=None,loss_type="pair",trainable=True):
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
        # self.lstm_keep_prob=lstm_keep_prob
       
        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [None,sequence_length], name="input_x_2")
        self.label=tf.placeholder(tf.float32, [None], name="label")
        self.lengths=tf.placeholder(tf.int32,[None],name='lengths')
        
        
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
           


        with tf.name_scope("prepare"):
            q  =tf.nn.embedding_lookup(self.Embedding_W, self.input_x_1)
            pos=tf.nn.embedding_lookup(self.Embedding_W, self.input_x_2[...,0])
            pos = tf.expand_dims(pos,1)
            self.q_pos = tf.concat([pos,q],1)

            
            # self.x12=self.data_concat(q,pos)
        


        fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size + 1,forget_bias=1.0)
        bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size + 1,forget_bias=1.0)
       

        with tf.variable_scope("dis"):
           
            outputs, states=tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,self.q_pos,sequence_length = self.lengths+1,dtype=tf.float32)
            
            states_fw, states_bw = states
            
            output_fw, output_bw = outputs
            self.lstm_output = tf.reduce_max(tf.concat([output_fw,output_bw],2),1)
            c_f, h_f = states_fw
            c_b, h_b= states_bw
            #可能还需要改动
            self.lstm_out=tf.concat([h_f,h_b],1)   


        with tf.variable_scope('MLP'):
            self.W1 = tf.get_variable('w1',[2*hidden_size+2,1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.b1 = tf.get_variable('b1',[1],initializer=tf.constant_initializer(0.0))
            #self.W2 = tf.get_variable('w2',[self.hidden_size,1],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            #self.score=tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(outputs12, self.W1, self.b1)), self.W2)
            self.score=tf.nn.xw_plus_b(self.lstm_output, self.W1, self.b1)
            self.score=tf.cast(tf.squeeze(self.score),tf.float32)
            self.updated_paras.append(self.W1)

            
        # self.l2_loss = tf.constant(0.0)
        # #给矩阵算一个平均值
        # for para in self.updated_paras:
        #     self.l2_loss+= tf.nn.l2_loss(para)


    def data_concat(self,q,a):
        
        # mulq=tf.multiply(q,a)
        # subq=tf.abs(tf.subtract(q,a))
        x=tf.concat([q,mulq,subq],2)


        return x

    # def attention(self,x):
    #     z = tf.reduce_sum(tf.contrib.layers.fully_connected(inputs=x,num_outputs=1,activation_fn=None),2)
    #     mulq=tf.multiply(q,a)
    #     subq=tf.abs(tf.subtract(q,a))

    #     attention = tf.nn.softmax(x)
    #     return attention


