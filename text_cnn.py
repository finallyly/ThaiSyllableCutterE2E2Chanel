#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2017 All rights reserved.
# 
#   FileName:CNN4NLP.py
#   Creator: yuliu1finally@gmail.com
#   Time:12/29/2017
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
import tensorflow as tf;
import numpy as np;
class TextCNN(object):

    def __init__(self,sequence_length,num_classes,dim,filter_sizes,num_filters,l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.float32,[sequence_length,dim,2],name="input_x");
        self.input_y = tf.placeholder(tf.float32,[sequence_length,num_classes],name="input_y");
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob");

        #keep track of l2 loss;
        l2_loss = tf.constant(0.0);

        #embedding layer
        with tf.name_scope("embedding"):
            self.embedding_chars_expanded = tf.expand_dims(self.input_x,0);

        #Create a convolution + maxpool layer for each filter size
        pooled_outputs = [];
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s"%filter_size):
                ksize=[filter_size,dim,2,num_filters];
                kernel = tf.Variable(tf.truncated_normal(ksize,stddev=0.1),name="kernel");
                b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b");
                conv = tf.nn.conv2d(self.embedding_chars_expanded,
                                    kernel,
                                    strides=[1,1,1,1],
                                    padding='VALID',
                                    name="conv");
                # Apply nonlnearity
                relu = tf.nn.relu(conv+b,name="relu");

                #Maxpooling over the outputs

                pooled = tf.nn.max_pool(relu,
                ksize=[1,sequence_length-filter_size+1,1,1],
                strides=[1,1,1,1],
                padding="VALID",
                name="pool");
                pooled_outputs.append(pooled);



        #Combine all the pooled features
        num_filters_total = num_filters*len(filter_sizes);
        self.h_pool = tf.concat(pooled_outputs,3);
        self.h_pool_flat = tf.reshape(self.h_pool,[sequence_length,30]);

        # Add dropout
        with tf.name_scope("droupout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob);
            self.shapeinfo = tf.shape(self.h_drop);

        with tf.name_scope("output"):
            W =tf.get_variable("weights",shape=[30,num_classes],initializer=tf.truncated_normal_initializer());
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]),name="b");
            l2_loss += tf.nn.l2_loss(W);
            l2_loss += tf.nn.l2_loss(b);
            self.scores = tf.matmul(self.h_drop,W)+b;
            logits = tf.nn.softmax(self.scores,name="logits");
            self.predictions=tf.argmax(logits,1,name="predictions");

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y);
            self.loss = tf.reduce_mean(losses)+l2_reg_lambda*l2_loss;

        #Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1));
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy");

