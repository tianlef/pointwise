import numpy as np
import traceback
import os
import time
import datetime
import operator
import random
import tensorflow as tf
import pickle
import copy




import json

import Discriminator
from data_helper import encode_sent
import data_helper

# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.05, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("hidden_size", 50, "Hidden Size (default: 100)")
tf.flags.DEFINE_integer("batch_size", 25, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()


timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))

print(("Loading data..."))

vocab = data_helper.build_vocab()  # 加载vocab,需修改参数 QA Web 默认是QA
embeddings =data_helper.load_vectors(vocab) #需修改参数 QA Web
alist = data_helper.read_alist("WebAP")  # 加载所有例子里的回答
qs=data_helper.qname("WebAP/qeo-train.txt")
number=data_helper.qid("WebAP/qeo-train.txt")
numbertest=data_helper.qid("WebAP/qeo-test.txt")
qstest=data_helper.qname("WebAP/qeo-test.txt")
test1List = data_helper.loadTestSet(dataset="WebAP",filename="term-test")
test2List = data_helper.loadTestSet(dataset="WebAP",filename="term-train")


print("Load done...")
precision = 'WebAP/log/test1.dns' + timeStamp
precision1 ='WebAP/logtrain/train.dns' + timeStamp
loss_precision = 'WebAP/log/qterm.gan_loss' + timeStamp

from functools import wraps


# print( tf.__version__)
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        #print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret

    return _deco

@log_time_delta
def generate_uniform():
    samples = []
    count=0
    for _index,i in enumerate(number):
        if i==0:
            continue
        neg_alist_index = [j for j in range(count,count+i)]

        pools=[]
        termpools=[] 
        #items中含有标签 问题长度 问题编号 问题 回答 
        for z in neg_alist_index:
            pools.append(alist[z])
        for pool in pools:
            termpools.append(pool[4])
        termpools=np.array(termpools)
        neg_index=[i for i in range(len(pools))]

        for  indexi,pair in enumerate(pools):
            label=pair[0]
            if int(label)==int(1):
                #print(indexi)
                neg_index.remove(indexi)
        for  pair in pools:
            label=pair[0]
            #print(pair)
            if int(label)==int(1) and len(neg_index) > 0:
                qq = pair[3]
                aa=pair[4]
                pos_len=pair[1]
                #print(pos_len)
                neg=termpools[neg_index]

                neg_samples = np.random.choice(neg, size=1, replace=False)
                #print(neg_samples)
                for neg in neg_samples:
                    samples.append([encode_sent(vocab, qq, FLAGS.max_sequence_length),
                    encode_sent(vocab, aa, FLAGS.max_sequence_length),
                    encode_sent(vocab, neg, FLAGS.max_sequence_length),
                    int(pos_len)
                    ])
                    
        count=count+i
    return samples

@log_time_delta
def test(sess, cnn, testList,merged):
    label = []
    sum_pair = 0
    for i in numbertest:
        sum_pair += i 
    for line in testList:
        label_item = int(line[0])
        label.append(label_item)
    
    x_test_1, x_test_2, x_test_3 = data_helper.load_val_batch(testList, vocab,0,sum_pair)
    feed_dict = {
        cnn.input_x_1: x_test_1,
        cnn.input_x_2: x_test_2,  # x_test_2 equals x_test_3 for the test case
        cnn.lengths:x_test_3,
        cnn.label : np.array(label)
        }
    summary,acc = sess.run([merged,cnn.accuracy], feed_dict)
    return summary,acc
 
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k] 
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k) 
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def dev_step(sess, cnn, testList):
    ndcg3=[]
    ndcg5=[]
    ndcg10=[]
    ndcg20=[]
    sum1=0
    for i in numbertest:
        queryList=[]
        x_test_1, x_test_2, x_test_3 = data_helper.load_val_batch(testList, vocab,sum1,i)
        feed_dict = {
            cnn.input_x_1: x_test_1,
            cnn.input_x_2: x_test_2,  # x_test_2 equals x_test_3 for the test case
            cnn.lengths:x_test_3
            }
        predicted = sess.run(cnn.score, feed_dict)
        for index,s in enumerate(predicted):
            line = testList[sum1+index]
            term=line[0]
            queryList.append((term, predicted[index]))
        sum1+=i
        queryList= sorted(queryList, key=lambda x: x[1])
        queryList.reverse()
        query_sort = [int(x[0]) for x in queryList]
        r = []
        for j in query_sort:
           r.append(j)

        ndcg3.append(ndcg_at_k(r, 3))
        ndcg5.append(ndcg_at_k(r, 5))
        ndcg10.append(ndcg_at_k(r, 10))
        ndcg20.append(ndcg_at_k(r, 20))


        return sum(ndcg3) * 1.0 / len(ndcg3),sum(ndcg5) * 1.0 / len(ndcg5),sum(ndcg10) * 1.0 / len(ndcg10),sum(ndcg20) * 1.0 / len(ndcg20)


@log_time_delta
def dev_step2(sess, cnn, testList):
    ndcg3=[]
    ndcg5=[]
    ndcg10=[]
    ndcg20=[]
    sum1=0
    for i in number:
        queryList=[]
        x_test_1, x_test_2, x_test_3 = data_helper.load_val_batch(testList, vocab,sum1,i)
        feed_dict = {
            cnn.input_x_1: x_test_1,
            cnn.input_x_2: x_test_2,# x_test_2 equals x_test_3 for the test case
            cnn.lengths:x_test_3
            }
        predicted = sess.run(cnn.score, feed_dict)       
        for index,s in enumerate(predicted):
            line = testList[sum1+index]
            term=line[0]
            queryList.append((term, predicted[index]))
        sum1+=i
        queryList= sorted(queryList, key=lambda x: x[1])
        queryList.reverse()
        query_sort = [int(x[0]) for x in queryList]
        r = []
        for j in query_sort:
           r.append(j)
        ndcg3.append(ndcg_at_k(r, 3))
        ndcg5.append(ndcg_at_k(r, 5))
        ndcg10.append(ndcg_at_k(r, 10))
        ndcg20.append(ndcg_at_k(r, 20))

        return sum(ndcg3) * 1.0 / len(ndcg3),sum(ndcg5) * 1.0 / len(ndcg5),sum(ndcg10) * 1.0 / len(ndcg10),sum(ndcg20) * 1.0 / len(ndcg20)

@log_time_delta
def evaluation(sess, model, log, num_epochs=0):
    current_step = tf.train.global_step(sess, model.global_step)
    if isinstance(model, Discriminator.Discriminator):
        model_type = "Dis"
    else:
        model_type = "Gen"
    now=time.time()
    local_time=time.localtime(now)
    this=str(time.strftime('%Y-%m-%d %H:%M:%S',local_time))
    ndcg3,ndcg5,ndcg10,ndcg20 = dev_step(sess, model, test1List)
    line = this+" type: %s test1: %d epoch: ndcg3 %f ndcg5 %f ndcg10 %f ndcg20 %f" % (model_type,current_step,ndcg3,ndcg5,ndcg10,ndcg20)
    print(line)
    #print(model.save_model(sess, ndcg3))
    log.write(line + "\n")
    log.flush()

def evaluation2(sess, model, log, num_epochs=0):
    current_step = tf.train.global_step(sess, model.global_step)
    if isinstance(model, Discriminator.Discriminator):
        model_type = "Dis"
    else:
        model_type = "Gen"
    now=time.time()
    local_time=time.localtime(now)
    this=str(time.strftime('%Y-%m-%d %H:%M:%S',local_time))
    

    ndcg3,ndcg5,ndcg10,ndcg20 = dev_step2(sess, model, test2List)
    line = this+" type: %s traintest1: %d epoch: ndcg3 %f ndcg5 %f ndcg10 %f ndcg20 %f" % (model_type,current_step,ndcg3,ndcg5,ndcg10,ndcg20)
    print(line)
    #print(model.save_model(sess, ndcg3))
    log.write(line + "\n")
    log.flush()
   


def main():
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default() ,open(precision,"w") as log,open(precision1,'w') as log1:
                 
                    param= None
                    loss_type = "point"
                    
                    discriminator = Discriminator.Discriminator(
                    sequence_length=FLAGS.max_sequence_length,
                    batch_size=FLAGS.batch_size,
                    vocab_size=len(vocab),
                    embedding_size=FLAGS.embedding_dim,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    embeddings=embeddings,
                    hidden_size=FLAGS.hidden_size,
                    paras=None,
                    loss_type=loss_type)

                    # saver = tf.train.Saver()    
                  
                    # merged = tf.summary.merge_all()
                    # train_writer = tf.summary.FileWriter('tensorboard_train',sess.graph)
                    # test_writer = tf.summary.FileWriter('tensorboard_test')
                    sess.run(tf.global_variables_initializer())
                   
                    for i in range(FLAGS.num_epochs):
                        
                        samples=generate_uniform() 
                        
                        for batch in data_helper.batch_iter(samples,batch_size=FLAGS.batch_size,
                            num_epochs=1,shuffle=True):  
                                  # try:
                            pred_data=[]
                            pred_data.extend(batch[:,1])
                            pred_data.extend(batch[:,2])
                            pred_data = np.asarray(pred_data)
                            pred_data_label=[]
                            pred_data_label = [1.0] * len(batch[:,1])
                            pred_data_label.extend([0.0] * len(batch[:,2]))
                            pred_data_label = np.asarray(pred_data_label)  
                            q = []
                            q.extend(batch[:,0])
                            q.extend(batch[:,0])
                            q = np.asarray(q)  
                            lengths=[]
                            lengths.extend(batch[:,3])
                            lengths.extend(batch[:,3])
                            lengths=np.asarray(lengths)
                            
                                
                            feed_dict = {
                                    discriminator.input_x_1: q,
                                    discriminator.input_x_2: pred_data,
                                    discriminator.label: pred_data_label,
                                    discriminator.lengths:lengths
                                }
                                
                                

                            _, accuracy,step, current_loss,score = sess.run(
                                    [discriminator.train_op,discriminator.accuracy,
                                    discriminator.global_step,
                                        discriminator.loss,discriminator.score],
                                    feed_dict)
                            # train_writer.add_summary(summary, i)
                                
                            time_str = datetime.datetime.now().isoformat()
                             
                            
                        # summary_test,accuracy_test = test(sess, discriminator, test1List,merged)
                        # print("test_epoch:"+str(i)+" accuracy:"+str(accuracy_test))
                        # test_writer.add_summary(summary_test,i)
                        if(i%3==0):
                            evaluation2(sess,discriminator,log1,i)
                            evaluation(sess,discriminator,log,i)
                            
                      

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        exstr=traceback.format_exc()
        print(repr(e))
        with open('error.txt', 'w') as f:
            f.write(exstr)
    else:
        print('c')
                 
