import tensorflow as tf;
import  numpy as np;
import os;
import time;
import datetime;
import TranData1;
from text_cnn import TextCNN;
from tensorflow.contrib import learn
import TranData1;
#Data loading params
tf.flags.DEFINE_float("dev_sample_percentage",0.001,"percentage of data to use for validation");
tf.flags.DEFINE_string("train_feature_file","./data/train.feature.txt","Data source for thai words in training");
tf.flags.DEFINE_string("train_label_file","./data/train.label.txt","Data source for label in training");
tf.flags.DEFINE_string("train_symbol_file","./data/train.symbol.txt","Data source for thai grammar symbols in training");
tf.flags.DEFINE_string("test_feature_file","./data/test.feature.txt","Data source for thai words in test");
tf.flags.DEFINE_string("test_label_file","./data/test.label.txt","Data source for label in test");
tf.flags.DEFINE_string("test_symbol_file","./data/test.symbol.txt","Data source for thai grammar symbols in test");

#Model Hyperparameters
tf.flags.DEFINE_integer("dim",100,"Dimensionality of character embedding (default: 100)");
tf.flags.DEFINE_integer("num_classes",8,"number of classes");
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 200)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

#Training parameters
tf.flags.DEFINE_integer("sequence_length", 20, "sequence_length (default: 20)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

#Check point file
tf.flags.DEFINE_string("checkpoint_dir", "/Users/liuyu/PycharmProjects/ThaiSyllableCNN2Chanel/runs/1515005011/checkpoints", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS;
FLAGS._parse_flags();
print("\nParameters:");

for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(),value));

print("");

print("Loading data...");

dataprocessor = TranData1.CorpusProcess(maxLen=FLAGS.sequence_length,dim=FLAGS.dim,numclasses=FLAGS.num_classes,dev_sample_percentage=FLAGS.dev_sample_percentage);
dataprocessor.load("data/vocab_all.pkl");
featureVec, labelVec, symbolVec = dataprocessor.FormatVec(FLAGS.test_feature_file,FLAGS.test_label_file,FLAGS.test_symbol_file);
X_test, Y_test, Xsymbol_test = dataprocessor.FormatVecForCNN(featureVec, labelVec, symbolVec);

# Evaluation
# ==========================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir);
graph = tf.Graph();
all_predictions=[];
fout=open("result.txt","w");
with graph.as_default():
    with tf.Session() as sess:
         #Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file));
        saver.restore(sess,checkpoint_file);
        input_x = graph.get_operation_by_name("input_x").outputs[0];
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0];

        #Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0];

        batches = dataprocessor.GetBatch(X_test, Y_test, Xsymbol_test);


        #Collect the predictions here
        for x_test_batch,y_test_batch in batches:
            batch_predictions = sess.run(predictions,{input_x:x_test_batch,dropout_keep_prob:1.0});
            #all_predictions = np.concatenate([all_predictions,batch_predictions]);
            newline="";
            for i in range(0,len(batch_predictions)):
                newline+=dataprocessor.label_id2vocab[batch_predictions[i]];
            fout.write("{:s}\n".format(newline));
fout.close();


