# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 04:04:47 2018

@author: tianle
"""

import tensorflow as tf 
import numpy as np 
import  pickle
import time
from score_gan import score
class Generator(score):
    
    def __init__(self, sequence_length,batch_size,vocab_size, embedding_size,hidden_size,l2_reg_lambda=0.0,paras=None,learning_rate=1e-2,embeddings=None,loss_type="pair",trainable=True):
        score.__init__(self,sequence_length,batch_size,vocab_size, embedding_size,hidden_size,l2_reg_lambda=l2_reg_lambda,paras=paras,learning_rate=learning_rate,embeddings=embeddings,loss_type=loss_type,trainable=trainable)
        self.model_type="Gen"
        self.reward  =tf.placeholder(tf.float32, shape=[None], name='reward')
        self.neg_index  =tf.placeholder(tf.int32, shape=[None], name='neg_index')


        self.batch_scores =tf.nn.softmax(self.score)
        self.prob = tf.gather(self.batch_scores,self.neg_index)
        # self.loss1=self.prob
        self.gan_loss =  -tf.reduce_mean(tf.log(self.prob) *self.reward) 
        #+l2_reg_lambda * self.l2_loss

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = optimizer.compute_gradients(self.gan_loss)
        
       

        self.capped_gvs = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -10., 10.), gv[1]], self.grads_and_vars)

        self.gan_updates = optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)

        