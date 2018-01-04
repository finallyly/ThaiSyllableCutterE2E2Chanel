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
tf.flags.DEFINE_integer("num_epochs", 1 , "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#Data Preparation

#=======================================================

#Load data
print("Loading data...");

dataprocessor = TranData1.CorpusProcess(maxLen=FLAGS.sequence_length,dim=FLAGS.dim,numclasses=FLAGS.num_classes,dev_sample_percentage=FLAGS.dev_sample_percentage);
dataprocessor.LoadTrainData(FLAGS.train_feature_file, FLAGS.train_label_file, FLAGS.train_symbol_file);
featureVec, labelVec, symbolVec = dataprocessor.FormatVec(FLAGS.train_feature_file,FLAGS.train_label_file,FLAGS.train_symbol_file);
featureVecShuffle, labelVecShuffle, symbolVecShuffle = dataprocessor.Shuffle(featureVec, labelVec, symbolVec);
del featureVec, labelVec, symbolVec;
train_featureVec, train_labelVec, \
train_symbolVec, dev_featureVec, \
dev_labelVec, dev_symbolVec = dataprocessor.Split(featureVecShuffle, labelVecShuffle, symbolVecShuffle);
X_train, Y_train, Xsymbol_train = dataprocessor.FormatVecForCNN(train_featureVec, train_labelVec, train_symbolVec);
X_dev, Y_dev, Xsymbol_dev = dataprocessor.FormatVecForCNN(dev_featureVec,dev_labelVec,dev_symbolVec);


dataprocessor.dumpPlain("data/train.feature.vid.txt", "data/train.feature.idv.txt", "data/train.label.vid.txt",
                        "data/train.label.idv.txt", "data/train.symbol.vid.txt", "data/train.symbol.idv.txt");
dataprocessor.dump("data/vocab_all.pkl");

print (np.array(X_train).shape);
print(np.array(Y_train).shape);
print(np.array(Xsymbol_train).shape);

with tf.Session() as sess:
    cnn=TextCNN(
        sequence_length=FLAGS.sequence_length,
        num_classes=FLAGS.num_classes,
        dim=FLAGS.dim,
        filter_sizes=list(map(int,FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda);
    sinfo=sess.run(cnn.shapeinfo);
    print "sinfo";
    print sinfo;
    global_step = tf.Variable(0,name="global_step",trainable=False);
    optimizer = tf.train.AdamOptimizer(1e-3);
    grads_and_vars = optimizer.compute_gradients(cnn.loss);
    train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step);

    grad_summaries = [];
    for g,v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name),g);
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g));
            grad_summaries.append(grad_hist_summary);
            grad_summaries.append(sparsity_summary);
    grad_summaries_merged = tf.summary.merge(grad_summaries);


    #Output directory for models and summaries
    timestamp = str(int(time.time()));
    out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp));
    print("Writng to {}\n".format(out_dir));

    #Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss",cnn.loss);
    acc_summary = tf.summary.scalar("accuracy",cnn.accuracy);


    #Train Summaries
    train_summary_op = tf.summary.merge([loss_summary,acc_summary,grad_summaries_merged]);
    train_summary_dir = os.path.join(out_dir,"summaries","train");
    train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph);

    #Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary,acc_summary]);
    dev_summary_dir = os.path.join(out_dir,"summaries","dev");
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir,sess.graph);

    #Checkpoint directory. Tensorflow assumes this directory already exists so we need to createit
    checkpoint_dir=os.path.abspath(os.path.join(out_dir,"checkpoints"));
    checkpoint_prefix = os.path.join(checkpoint_dir,"model");
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir);
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints);


    #Initialize all variables
    sess.run(tf.global_variables_initializer());

    def train_step(x_batch,y_batch,index):
        """
        A single training step
        :param x_batch:
        :param y_batch:
        :return:
        """
        feed_dict = {
            cnn.input_x:x_batch,
            cnn.input_y:y_batch,
            cnn.dropout_keep_prob:1.0
        }
        _, step,summaries,loss,accuracy = sess.run(
        [train_op,global_step,train_summary_op,cnn.loss,cnn.accuracy],
        feed_dict);
        time_str = datetime.datetime.now().isoformat();
        if index %1000 ==0:
            print("{}:step {},loss{:g},acc{:g}".format(time_str,step,loss,accuracy));
        train_summary_writer.add_summary(summaries,step);


    def dev_step(x_batch,y_batch,index,writer=None):
        """
        :param x_batch:
        :param y_batch:
        :param writer:
        :return:
        """
        feed_dict = {
            cnn.input_x:x_batch,
            cnn.input_y:y_batch,
            cnn.dropout_keep_prob:1.0
        };
        step, summaries, loss,accuracy = sess.run(
            [global_step,dev_summary_op,cnn.loss,cnn.accuracy],
           feed_dict );
        time_str = datetime.datetime.now().isoformat();
        if index%10==0:
            print("{}:step {},loss {:g}, acc{:g}".format(time_str,step,loss,accuracy));
        if writer:
            writer.add_summary(summaries,step);


    #Generate batches
    for i in range(FLAGS.num_epochs):
        trainIndex=0;
        batches = dataprocessor.GetBatch(X_train,Y_train,Xsymbol_train);
        for x_batch,y_batch in batches:
            trainIndex+=1;
            train_step(x_batch,y_batch,trainIndex);
            current_step = tf.train.global_step(sess,global_step);
            if current_step% FLAGS.evaluate_every == 0:
                print("\n Evaluation:");
                devbatches = dataprocessor.GetBatch(X_dev,Y_dev,Xsymbol_dev);
                index = 0;
                for xdev_batch,ydev_batch in devbatches:
                    index+=1;
                    dev_step(xdev_batch, ydev_batch, index,writer=dev_summary_writer);
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess,checkpoint_prefix, global_step=current_step);
                print("Saved model checkpoint to {}\n".format(path));