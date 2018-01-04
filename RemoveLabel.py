#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2018 All rights reserved.
# 
#   FileName:RemoveLabel.py
#   Creator: yuliu1finally@gmail.com
#   Time:01/03/2018
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
with open("result.txt","r") as f:
    with open("result.filter.txt","w") as fout:
        for line in f:
            line = line.strip();
            padding_index = line.find("<PAD>");
            newline="";
            if padding_index!=-1:
                newline=line[:padding_index];
            else:
                newline = line;
            fout.write("%s\n"%newline);
        
