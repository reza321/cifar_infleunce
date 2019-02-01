import tensorflow as tf
import numpy as np
import os
import math
import sys
import pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import cPickle
import gzip
import urllib.request
from tensorflow.contrib.learn.python.learn.datasets import base
from dataset import DataSet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay


def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        # lab = np.identity(10)[arr[0]]
        lab = arr[0]        

        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)    # To get a good picture comment this and pass only img
        
    return np.array(images),np.array(labels)

def load_cifar():
    train_data = []
    train_labels = []
    
    if not os.path.exists("cifar-10-batches-bin"):
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                   "cifar-data.tar.gz")
        os.popen("tar -xzf cifar-data.tar.gz").read()
        

    for i in range(5):
        r,s = load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
        train_data.extend(r)
        train_labels.extend(s)
        
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    test_data, test_labels = load_batch("cifar-10-batches-bin/test_batch.bin")

    test_data=test_data
    train_data=train_data
    # print(train_data.shape)
    # plt.imshow(test_data[30])
    # plt.show()
    # test_data=test_data.reshape(test_data.shape[0],-1)
    # print(train_data[30].shape)
    # print((train_data[30].dtype))
    # test_data=test_data.reshape(test_data.shape[0],32,32,3).astype(np.uint8)
    # plt.imshow(test_data[30])
    # plt.show()
    # exit()
    VALIDATION_SIZE = 5000
    validation_data = train_data[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]

    train_data = train_data[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    train = DataSet(train_data, train_labels)
    validation = DataSet(validation_data, validation_labels)
    test = DataSet(test_data, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)

class CIFARModel(GenericNeuralNet):
    def __init__(self,input_channels,weight_decay,input_side,dense1_unit,dense2_unit,kernel_size,filter1,filter2,**kwargs):
        self.weight_decay = weight_decay
        self.input_channels = input_channels
        self.input_side = input_side
        self.input_dim = self.input_side * self.input_side * self.input_channels
        self.kernel_size=kernel_size
        self.filter1=filter1
        self.filter2=filter2
        self.dense1_unit=dense1_unit
        self.dense2_unit=dense2_unit

        

        super(CIFARModel, self).__init__(**kwargs)
    def conv2d_maker(self, input_x,kernel_size, input_channels,filter_size, stride):
        strides=[1, stride, stride, 1]
        weights = variable_with_weight_decay(
            'weights', 
            [kernel_size *kernel_size * input_channels * filter_size],
            stddev=2.0 / math.sqrt(float(kernel_size* kernel_size * input_channels)),
            wd=self.weight_decay)
        biases = variable(
            'biases',
            [filter_size],
            tf.constant_initializer(0.0))

        weights_reshaped = tf.reshape(weights, [kernel_size,kernel_size, input_channels, filter_size])        
        hidden=tf.nn.relu(tf.nn.conv2d(input_x, weights_reshaped, strides=strides,padding='VALID')+biases)
        return hidden
#    def inference_orig(self,input_x):
        # input_x_reshaped=np.reshape(input_x,(input_x.shape[0],self.input_side,self.input_side,self.input_channels))
#        input_x_reshaped=tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])
#        conv1=tf.layers.conv2d(inputs=input_x_reshaped,filters=64,kernel_size=3,activation=tf.nn.relu,padding='valid',name='conv1')
#        conv2=tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=3,activation=tf.nn.relu,padding='valid',name='conv2')
#        pool1=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=1,name='pool1')#

#        conv3=tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=3,activation=tf.nn.relu,padding='valid',name='conv3')
#        conv4=tf.layers.conv2d(inputs=conv3,filters=128,kernel_size=3,activation=tf.nn.relu,padding='valid',name='conv4')
#        pool2=tf.layers.max_pooling2d(inputs=conv4,pool_size=[2,2],strides=1,name='pool2')     

#        flat1=tf.layers.flatten(pool2)
#        dense1=tf.layers.dense(inputs=flat1,units=256,activation=tf.nn.relu,name='dense1')
#        dropout=tf.layers.dropout(inputs=dense1,rate=0.5)
#        dense2=tf.layers.dense(inputs=dropout,units=256,activation=tf.nn.relu,name='dense2')
#        logits=tf.layers.dense(inputs=dense2,units=10,name='logits')
#        return logits

    def inference(self,input_x):
        # input_x_reshaped=np.reshape(input_x,(input_x.shape[0],self.input_side,self.input_side,self.input_channels))
        
        input_x_reshaped=tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])
        
        with tf.variable_scope('conv1'):
            conv1 = self.conv2d_maker(input_x_reshaped, self.kernel_size, self.input_channels, self.filter1, stride=1)

        with tf.variable_scope('conv2'):
            conv2 = self.conv2d_maker(conv1, self.kernel_size, self.filter1, self.filter1, stride=1)

        pool1=tf.nn.max_pool(value=conv2,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID',name='pool1')
        
        with tf.variable_scope('conv3'):
            conv3 = self.conv2d_maker(pool1, self.kernel_size, self.filter1, self.filter2, stride=1)
        
        with tf.variable_scope('conv4'):
            conv4 = self.conv2d_maker(conv3, self.kernel_size, self.filter2, self.filter2, stride=1)
        
        pool2=tf.nn.max_pool(value=conv4,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID',name='pool2')

        flat1=tf.contrib.layers.flatten(pool2)

        
        with tf.variable_scope('dense1'):
            weights = variable_with_weight_decay('weights', [flat1.get_shape().as_list()[-1]* self.dense1_unit],stddev=1.0 / math.sqrt(float(self.dense1_unit)),wd=self.weight_decay)            
            biases = variable('biases',[self.dense1_unit],tf.constant_initializer(0.0))

            dense1 = tf.nn.relu(tf.matmul(flat1, tf.reshape(weights, [flat1.get_shape().as_list()[-1], self.dense1_unit])) + biases)

        dropout=tf.layers.dropout(dense1,rate=0.5)

        with tf.variable_scope('dense2'):
            weights = variable_with_weight_decay('weights', [self.dense1_unit * self.dense2_unit],stddev=1.0 / math.sqrt(float(self.dense2_unit)),wd=self.weight_decay)            
            biases = variable('biases',[self.dense2_unit],tf.constant_initializer(0.0))

            dense2 = tf.nn.relu(tf.matmul(dropout, tf.reshape(weights, [self.dense1_unit, self.dense2_unit])) + biases)

        logits_unit=10
        with tf.variable_scope('logits'):
            weights = variable_with_weight_decay('weights', [self.dense2_unit * logits_unit],stddev=1.0 / math.sqrt(float(logits_unit)),wd=self.weight_decay)            
            biases = variable('biases',[logits_unit],tf.constant_initializer(0.0))

            logits = tf.matmul(dense2, tf.reshape(weights, [self.dense2_unit, logits_unit])) + biases

        return logits


    def load_trained_model(self, input_x):
            model = Sequential()           
            
            model=self.inference(input_x)

            model.load_weights( self.file_name)
            return model            

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds 


    def retrain(self, num_steps, feed_dict):        

        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])
        
        for step in xrange(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)


    def get_all_params(self):
        # names=[n.name for n in tf.get_default_graph().as_graph_def().node]
        all_params = []
        for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'dense1', 'dense2','logits']:        
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)

        return all_params 


    # def get_all_params_sample(self):
    #     names = [weight.name for layer in self.model.layers for weight in layer.weights]
    #     weights = self.model.get_weights()
    #     for name, weight in zip(names, weights):
    #         print(name, weight.shape)


    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder







