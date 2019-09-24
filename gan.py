# coding=utf-8
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
import Generator
from data_helper import encode_sent
import data_helper

# import dataHelper   
# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")



tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.05, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("hidden_size",50, "Hidden Size (default: 100)")
tf.flags.DEFINE_integer("batch_size", 25, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("pools_size", 50, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("gen_pools_size", 20, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("g_epochs_num", 2, " the num_epochs of generator per epoch")
tf.flags.DEFINE_integer("d_epochs_num", 5, " the num_epochs of discriminator per epoch")
tf.flags.DEFINE_integer("sampled_size", 100, " the real selectd set from the The sampled pools")
tf.flags.DEFINE_integer("sampled_temperature", 0.5, " the temperature of sampling")
tf.flags.DEFINE_integer("gan_k", 10, "he number of samples of gan")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()


timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))

print(("Loading data..."))

vocab = data_helper.build_vocab()  
embeddings = data_helper.load_vectors(vocab) 
alist = data_helper.read_alist("WebAP")  
raw = data_helper.read_raw("WebAP")  
qs = data_helper.qname("WebAP/qeo-train.txt")
number = data_helper.qid("WebAP/qeo-train.txt")

numbertest = data_helper.qid("WebAP/qeo-test.txt")
qstest = data_helper.qname("WebAP/qeo-test.txt")
# 加载了几个测试集
test1List = data_helper.loadTestSet(dataset="WebAP",filename="term-test")

print("Load done...")
log_precision = 'WebAP/log/qterm.gan_precision' + timeStamp
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
        print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret

    return _deco


def generate_gan(sess, model, loss_type="point"):
    samples = []
    count=0
    for index,i in enumerate(number):
        if i == 0:
            continue
        neg_alist_index = [j for j in range(count,count+i)]
        pools = []
        termpools = []
        pos_term = []
        length = []
        query = ''
        flag = 0
        for neg in neg_alist_index:
            pools.append(alist[neg])
        for pool in pools:
            termpools.append(pool[4])
        termpools=np.array(termpools)

        for  pair in pools:
            if flag == 0:
                query = pair[3]
                flag = 1
            label=pair[0]
            length.append(int(pair[1]))
            if int(label) == int(1):
                pos=pair[4]
                pos_term.append(pos)
                
        candidata_list_score = []
        canditates = data_helper.loadCandidateSamples(query,termpools,vocab,length)
        for batch in data_helper.batch_iter(canditates, batch_size=FLAGS.batch_size):
            feed_dict = {model.input_x_1: np.array(batch[:,0]), 
                        model.input_x_2: np.array(batch[:,1]),
                        model.length: np.array(batch[:2])}
            predicated = sess.run(model.gan_score, feed_dict)
            candidate_list_score.extend(predicated)
        #softmax for candicate
        #exp_rating = np.exp(np.array(predicteds) * FLAGS.sampled_temperature * 1.5)
        exp_rating = np.exp(np.array(candidata_list_score) - np.max(candidate_list_score))
        prob = exp_rating / np.sum(exp_rating)
        prob = np.reshape(prob, len(prob))
        neg_samples = np.random.choice(termpools, size=pos_number, p=prob, replace=False)
        # for neg in neg_samples:
        #     samples.append([encode_sent(vocab, item, FLAGS.max_sequence_length) for item in [qq, aa, neg]])
        for j in pos_term:
            samples.append((
            encode_sent(vocab,query,FLAGS.max_sequence_length),
            encode_sent(vocab,pos_term[j],FLAGS.max_sequence_length),
            encode_sent(vocab,neg_samples[j],FLAGS.max_sequence_length),
            int(length[0])
            ))
        count=count+i
        random.shuffle(samples)
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

@log_time_delta
def dev_step1(sess, cnn, testList):
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
def evaluation1(sess, model, log, num_epochs=0):
    current_step = tf.train.global_step(sess, model.global_step)
    if isinstance(model, Discriminator.Discriminator):
        model_type = "Dis"
    else:
        model_type = "Gen"
    now=time.time()
    local_time=time.localtime(now)
    this=str(time.strftime('%Y-%m-%d %H:%M:%S',local_time))
    ndcg3,ndcg5,ndcg10,ndcg20 = dev_step1(sess, model, test1List)
    line = this+" type: %s test1: %d epoch: ndcg3 %f ndcg5 %f ndcg10 %f ndcg20 %f" % (model_type,current_step,ndcg3,ndcg5,ndcg10,ndcg20)
    print(line)
    #print(model.save_model(sess, ndcg3))
    log.write(line + "\n")
    log.flush()

@log_time_delta
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
        with tf.device("/gpu:1"):
            gpu_options = tf.GPUOptions(allow_growth=True)
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=gpu_options)
            sess = tf.Session(config=session_conf)

            with sess.as_default(), open(log_precision, "w") as log, open(loss_precision, "w") as loss_log:
                
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
                    loss=loss_type)

                generator = Generator.Generator(
                    sequence_length=FLAGS.max_sequence_length,
                    batch_size=FLAGS.batch_size,
                    vocab_size=len(vocab),
                    embedding_size=FLAGS.embedding_dim,
                    learning_rate=FLAGS.learning_rate * 0.1,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    embeddings=embeddings,
                    hidden_size=FLAGS.hidden_size,
                    paras=None,
                    loss=loss_type)

                sess.run(tf.global_variables_initializer())
                # evaluation(sess,discriminator,log,0)
                for i in range(FLAGS.num_epochs):
                    if i >= 0:
                        for j in range(FLAGS.d_epochs_num):
                            #G generate for d 
                            samples = generate_gan(sess, generator)
                            for _index, batch in enumerate(
                                data_helper.batch_iter(samples, 
                                num_epochs=FLAGS.d_epochs_num,
                                batch_size=FLAGS.batch_size,
                                shuffle=True)): 
                                #data processing
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
                                _, step, current_loss, accuracy = sess.run(
                                    [discriminator.train_op, 
                                     discriminator.global_step, 
                                     discriminator.loss,
                                     discriminator.accuracy],
                                    feed_dict)
                                line = ("%s: DIS step %d, loss %f with acc %f " % (
                                datetime.datetime.now().isoformat(), step, current_loss, accuracy))
                                if _index % 5==0:
                                    print(line)
                                loss_log.write(line+"\n")
                                loss_log.flush()

                        evaluation(sess, discriminator, log, i)

                    # 我们先对生成模型迭代g_epochs_num                    
                    for g_epoch in range(FLAGS.g_epochs_num):
                        count = 0
                        # _index = 0
                        samples = []
                        for index,i in enumerate(number):

                            pools=[]
                            termpools=[]
                            pos_term=[]
                            length=[]

                            if i == 0:
                                continue
                            neg_alist_index = [j for j in range(count,count+i)]
                            
                            for neg in neg_alist_index:
                                pools.append(alist[neg])
                            for pool in pools:
                                termpools.append(pool[4])
                            # termpools=np.array(termpools)

                            for  pair in pools:
                                label = int(pair[0])
                                length.append(pair[1])
                                
                                if label == int(1):
                                    query = pair[3]
                                    pos=pair[4]
                                    pos_term.append(pos)

                            candidate_list_score = []
                            candidates = data_helper.loadCandidateSamples(query,termpools,vocab,length)
                            
                            for batch in data_helper.batch_iter(candidates, batch_size=FLAGS.batch_size):
                                
                                feed_dict = {
                                    generator.input_x_1: np.asarray(batch[:0]),
                                    generator.input_x_2: np.asarray(batch[:1]),
                                    generator.lengths:np.asarray(batch[:2])
                                }
                                predicated = sess.run(model.gan_score, feed_dict)
                                candidate_list_score.extend(predicated)

                            exp_rating = np.exp(np.array(candidata_list_score) - np.max(candidate_list_score))

                            prob = exp_rating / np.sum(exp_rating)
                            prob = np.reshape(prob, len(prob))
                            pn = (1-FLAGs.sampled_temperature) * prob

                            for p,pair in enumerate(pools):
                                if if int(label) == int(1):
                                    pn[p] += FLAGS.sampled_temperature * 1.0 / len(pos_term) 

                            choose_index = np.random.choice(np.arange(len(pools)), [5 * len(pos_term)], p=pn)                                                            
                            subsamples = data_helper.loadCandidateSamples(query,termpools[choose_index], vocab,length[choose_index])
                            ###########################################################################
                            # Get reward and adapt it with importance sampling
                            ###########################################################################
                            feed_dict = {discriminator.input_x_1: np.asarray(subsamples[:, 0]),
                                        discriminator.input_x_2: np.asarray(subsamples[:, 1]),
                                        discriminator.length: np.asarray(subsamples[:, 2])}

                            reward = sess.run(discriminator.reward, feed_dict)  
                            reward = np.reshape(reward, len(reward)) 

                            feed_dict = {generator.input_x_1: np.asarray(canditates[:, 0]),
                                        generator.input_x_2: np.asarray(canditates[:, 1]),
                                        generator.neg_index: choose_index,
                                        generator.length: np.asarray(canditates[:, 2]),
                                        generator.reward: reward
                                        }

                            _, step, current_loss,score= sess.run(  
                                            [generator.gan_updates, 
                                            generator.global_step, 
                                            generator.gan_loss,  
                                            generator.score], 
                                            feed_dict)  

                            line = ("%s: GEN step %d, loss %f  score %f " % (
                                        datetime.datetime.now().isoformat(), step, current_loss, score))
                            if _index % 10==0:
                                print(line)
                            loss_log.write(line+"\n")
                            loss_log.flush()
                            count=count+i
                        dev_step1(sess,  generator, test1List,i*FLAGS.g_epochs_num + g_epoch)
                        evaluation(sess, generator, log, i*FLAGS.g_epochs_num + g_epoch)
                        log.flush()
                    


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


