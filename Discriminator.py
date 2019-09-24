# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:36:25 2018

@author: tianle
"""


import tensorflow as tf 
import numpy as np 
import time
import pickle
from score import score          
class Discriminator(score):
  

    def __init__(self,  sequence_length,batch_size,vocab_size, embedding_size,hidden_size,l2_reg_lambda=0.0,learning_rate=1e-2,paras=None,embeddings=None,loss_type="pair",trainable=True):
        
        score.__init__(self, sequence_length, batch_size,vocab_size, embedding_size,hidden_size,l2_reg_lambda=l2_reg_lambda,paras=paras,learning_rate=learning_rate,embeddings=embeddings,loss_type=loss_type,trainable=trainable)
        self.model_type="Dis"

    
        with tf.name_scope("output"):

            self.loss = tf.cast(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.label)),tf.float32)
            
            self.reward = (tf.sigmoid(self.score) - 0.5) * 2

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            # self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope("accuary"):
            self.predict = tf.nn.sigmoid(self.score)
           
            self.correct = tf.round(tf.abs(self.predict - self.label))
            self.correct1 = 1 - self.correct
            
          
            self.accuracy = 1 - tf.reduce_sum(self.correct)/(tf.reduce_sum(self.correct)+tf.reduce_sum(self.correct1))
        tf.summary.scalar('accuracy', self.accuracy)
                

      


            


