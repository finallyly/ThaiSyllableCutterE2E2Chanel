#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2017 All rights reserved.
# 
#   FileName:DataProcessor.py
#   Creator: yuliu1finally@gmail.com
#   Time:12/26/2017
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
from collections import Counter;
import numpy as np;
import cPickle;
class CorpusProcess(object):
    def __init__(self,maxLen=20,dim=100,numclasses=8,dev_sample_percentage=0.01,batch_size=5):
        self.feature_vocab2id = dict();
        self.feature_id2vocab = dict();
        self.label_vocab2id = dict();
        self.label_id2vocab = dict();
        self.symbol_vocab2id = dict();
        self.symbol_id2vocab = dict();
        self.maxLen = maxLen;
        self.dim = dim;
        self.numclasses=numclasses;
        self.dev_sample_percentage=dev_sample_percentage;
        self.batch_size = batch_size;

    def Process1(self,infile,ofile1,ofile2,ofile3):
        fout1 = open(ofile1,"w");
        fout2 = open(ofile2,"w");
        fout3 = open(ofile3,"w");
        newline1 = "";
        newline2 = "";
        newline3 = "";
        with open(infile,"r") as fid:
            for line in fid:
                line = line.strip();
                col = line.split("\t");
                if len(col)!=3:
                    if(newline1 != "" and newline2 != "" and newline3 != ""):
                        fout1.write("%s\n"%newline1);
                        fout2.write("%s\n"%newline2);
                        fout3.write("{:s}\n".format(newline3));
                        newline1="";
                        newline2="";
                        newline3="";
                    else:
                        newline1+=col[0];
                        newline2+=col[2];
                    if newline3!="":
                        newline3+="#";
                        newline3+=col[1];
        fout1.close();
        fout2.close();
        fout3.close();


    def LoadTrainData(self,infile1,infile2,infile3):
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        features = [];
        feature_words = [];
        labels = [];
        label_words = [];
        symbols = [];
        symbol_words = [];

        with open(infile1,"r") as f:
            for line in f:
                line = line.strip();
                uline = line.decode("UTF-8");
                features.append(line);
                for m in uline:
                    feature_words.append(m.encode("UTF-8"));

        with open(infile2,"r") as f:
            for line in f:
                line = line.strip();
                uline=line.decode("UTF-8");
                labels.append(line);
                for m in uline:
                    label_words.append(m.encode("UTF-8"));

        with open(infile3,"r") as f:
            for line in f:
                line = line.strip();
                symbols.append(line);
                col = line.split("#");
                for m in col:
                    symbol_words.append(m);


        counts_feature = Counter(feature_words);
        vocab_feature = sorted(counts_feature,key=counts_feature.get,reverse=True);
        self.feature_vocab2id = {word:ii for ii,word in enumerate(vocab_feature+special_words)};
        self.feature_id2vocab = {ii:word for word, ii in self.feature_vocab2id.iteritems()};


        counts_label = Counter(label_words);
        vocab_label = sorted(counts_label,key=counts_label.get,reverse=True);
        self.label_vocab2id = {word:ii for ii, word in enumerate(vocab_label+special_words)};
        self.label_id2vocab = {ii:word for word, ii in self.label_vocab2id.iteritems()};

        counts_symbol = Counter(symbol_words);
        vocab_symbol  = sorted(counts_symbol,key=counts_symbol.get,reverse=True);
        self.symbol_vocab2id = {word:ii for ii, word in enumerate(vocab_symbol+special_words)};
        self.symbol_id2vocab = {ii:word for word, ii in self.symbol_vocab2id.iteritems()};

    def dump(self,filename):
        cPickle.dump((self.feature_vocab2id,self.feature_id2vocab,
                      self.label_vocab2id,self.label_id2vocab,self.symbol_vocab2id,self.symbol_id2vocab),open(filename,"wb"));
    def load(self,filename):
        (self.feature_vocab2id,self.feature_id2vocab,
        self.label_vocab2id,self.label_id2vocab,
        self.symbol_vocab2id,self.symbol_id2vocab) = cPickle.load(open(filename,"rb"));
        print "feature_vocab2id{:d},feature_id2vocab{:d},label_vocab2id{:d}," \
              "label_id2vocab{:d},symbol_vocab2id{:d},symbol_id2vocab{:d}".format(len(self.feature_vocab2id),len(self.feature_id2vocab),
                                                                                  len(self.label_vocab2id),len(self.label_id2vocab),
                                                                                  len(self.symbol_vocab2id),len(self.symbol_id2vocab));




    def dumpPlain(self,ofile1,ofile2,ofile3,ofile4,ofile5,ofile6):
        with open(ofile1,"w") as f:
            for item in self.feature_vocab2id.iteritems():
                f.write("{:s}\t{:d}\n".format(item[0],item[1]));

        with open(ofile2,"w") as f:
            for item in self.feature_id2vocab.iteritems():
                f.write("{:d}\t{:s}\n".format(item[0],item[1]));

        with open(ofile3,"w") as f:
            for item in self.label_vocab2id.iteritems():
                f.write("{:s}\t{:d}\n".format(item[0],item[1]));
        with open(ofile4,"w") as f:
            for item in self.label_id2vocab.iteritems():
                f.write("{:d}\t{:s}\n".format(item[0], item[1]));
        with open(ofile5,"w") as f:
            for item in self.symbol_vocab2id.iteritems():
                f.write("{:s}\t{:d}\n".format(item[0], item[1]));
        with open(ofile6,"w") as f:
            for item in self.symbol_id2vocab.iteritems():
                f.write("{:d}\t{:s}\n".format(item[0], item[1]));




    def FormatVec(self,infile1,infile2,infile3):
        features=[];
        labels =[];
        symbols = [];
        with open(infile1,"r") as f:
            for line in f:
                line = line.strip();
                uline = line.decode("UTF-8");
                features.append([self.feature_vocab2id.get(uword.encode("UTF-8"),self.feature_vocab2id["<UNK>"]) for uword in uline]);

        with open(infile2,"r") as f:
            for line in f:
                line = line.strip();
                uline = line.decode("UTF-8");
                labels.append([self.label_vocab2id.get(uword.encode("UTF-8"),self.label_vocab2id["<UNK>"]) for uword in uline]);

        with open(infile3,"r") as f:
            for line in f:
                line = line.strip();
                col = line.split("#");
                symbols.append([self.symbol_vocab2id.get(elem,self.symbol_vocab2id["<UNK>"]) for elem in col]);


        featureVecPad=[feature+[self.feature_vocab2id["<PAD>"]]*(self.maxLen-len(feature)) for feature in features];
        labelVecPad=[label+[self.label_vocab2id["<PAD>"]]*(self.maxLen-len(label)) for label in labels];
        symbolVecPad=[symbol+[self.symbol_vocab2id["<PAD>"]]*(self.maxLen-len(symbol)) for symbol in symbols];
        return featureVecPad,labelVecPad,symbolVecPad;

    def Shuffle(self,featureVecPad,labelVecPad,symbolVecPad):
        np.random.seed(10);
        shuffle_indices = np.random.permutation(len(labelVecPad));
        featureVec_shuffled = np.array(featureVecPad)[shuffle_indices];
        labelVec_shuffled = np.array(labelVecPad)[shuffle_indices];
        symbolVec_shuffled = np.array(symbolVecPad)[shuffle_indices];
        return featureVec_shuffled.tolist(),labelVec_shuffled.tolist(),symbolVec_shuffled.tolist();

    def Split(self,featureVec,labelVec,symbolVec):
        dev_sample_index = -1 * int(self.dev_sample_percentage * float(len(featureVec)));
        train_featureVec=featureVec[:dev_sample_index];
        train_labelVec=labelVec[:dev_sample_index];
        train_symbolVec=symbolVec[:dev_sample_index];
        dev_featureVec=featureVec[dev_sample_index:];
        dev_labelVec=labelVec[dev_sample_index:];
        dev_symbolVec=symbolVec[dev_sample_index:];
        return train_featureVec,train_labelVec,train_symbolVec,dev_featureVec,dev_labelVec,dev_symbolVec;






    def FormatVecForCNN(self,featureVecPad,labelVecPad,symbolVecPad):
        x_feature=[];
        y=[];
        x_symbol=[];
        for i in range(0,len(featureVecPad)):
            temp=[];
            for j in range(0,len(featureVecPad[i])):
                x=[0.0]*self.dim;
                x[featureVecPad[i][j]]=1.0;
                temp.append(x);
            x_feature.append(temp);

        for i in range(0,len(symbolVecPad)):
            temp=[];
            for j in range(0,len(symbolVecPad[i])):
                x=[0.0]*self.dim;
                x[symbolVecPad[i][j]]=1.0;
                temp.append(x);
            x_symbol.append(temp);

        for i in range(0,len(labelVecPad)):
            temp=[];
            for j in range(0,len(labelVecPad[i])):
                x=[0.0]*self.numclasses;
                x[labelVecPad[i][j]]=1.0;
                temp.append(x);
            y.append(temp);
        return x_feature,y,x_symbol;

    def ReshapeMat(self,x,xs):
        merged = [];
        for i in range(0,len(x)):
            t=[];
            for j in range(0,len(x[0])):
                t.append([x[i][j],xs[i][j]]);
            merged.append(t);
        return merged;


    def GetBatch(self,x,y,xs):
        for i in range(0,len(x)):
            merged = self.ReshapeMat(x[i], xs[i]);
            yield np.array(merged),np.array(y[i]);






















def main():
    dataprocessor=CorpusProcess();
    dataprocessor.LoadTrainData("data/train.feature.txt", "data/train.label.txt", "data/train.symbol.txt");
    featureVec,labelVec,symbolVec=dataprocessor.FormatVec("data/train.feature.txt","data/train.label.txt","data/train.symbol.txt");
    featureVecShuffle,labelVecShuffle,symbolVecShuffle=dataprocessor.Shuffle(featureVec,labelVec,symbolVec);
    del featureVec,labelVec,symbolVec;
    train_featureVec, train_labelVec, \
    train_symbolVec, dev_featureVec, \
    dev_labelVec, dev_symbolVec=dataprocessor.Split(featureVecShuffle,labelVecShuffle,symbolVecShuffle);
    X_train,Y_train,Xsymbol_train=dataprocessor.FormatVecForCNN(train_featureVec,train_labelVec,train_symbolVec);
    print (np.array(X_train).shape);
    print(np.array(Y_train).shape);
    print(np.array(Xsymbol_train).shape);
    dataprocessor.ReshapeMat(X_train[0],Xsymbol_train[0]);

    dataprocessor.dumpPlain("data/train.feature.vid.txt","data/train.feature.idv.txt","data/train.label.vid.txt","data/train.label.idv.txt","data/train.symbol.vid.txt","data/train.symbol.idv.txt");
    dataprocessor.dump("data/vocab_all.pkl");
    dataprocessor.load("data/vocab_all.pkl");





    



    #print(len(featureVecPad));
    #print(len(featureVecPad[0]));
    #print(len(labelVecPad));
    #print(len(labelVecPad[0]));
    #print(len(symbolVecPad));
    #print(len(symbolVecPad[0]));
    #print(featureVecPad[0]);
    #print(labelVecPad[0]);
    #print(symbolVecPad[0]);



if __name__=="__main__":
    main();


