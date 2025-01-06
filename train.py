import os
import time
import pickle
import argparse

import numpy as np

from model import *
import utiles

np.random.seed(618)

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--model', type=str, default="hsskipgram", help="write model name")
parser.add_argument('--file_num', type=int, default=3, help='Number of files parameter')
parser.add_argument('--hidden_size', type=int, default=50, help='Hidden size parameter')
parser.add_argument('--max_epoch', type=int, default=1, help='max_epoch')
parser.add_argument('--print_point', type=int, default=1000000, help='How frequently do you want to print intermediate results?')
parser.add_argument('--threshold', type=int, default=-1, help='How many vocabulary do you want')
parser.add_argument('--window', type=int, default=5, help='window size')
args = parser.parse_args()

file_num = args.file_num
if file_num == 0:
    root_path = "input your path"
    path_list = [root_path + "/" + path for path in os.listdir(root_path)]
    root_path = "input your path"
    path_list += [root_path + "/" + path for path in os.listdir(root_path)]
    file_num = len(path_list)
    thres = 1000000
else :
    root_path = "input your path"
    path_list = [root_path + "/" + path for path in os.listdir(root_path)[:file_num]]
    thres = args.threshold

print("Threshold : ",thres)
word2id, id2vocab = utiles.vocabulary(path_list, Threshold=thres)
ht = utiles.Huffman_Tree(id2vocab)
ht.create_tree()

vocab_size = len(word2id)
hidden_size = args.hidden_size
window_size = args.window
max_epoch = args.max_epoch
start_lr = 0.025
lr = start_lr

model_name = args.model
if model_name == "hscbow":
    model = HS_CBOW(vocab_size, hidden_size, id2vocab)
    print("HS CBOW STANDBY")
elif model_name == "hsskipgram":
    model = HS_Skipgram(vocab_size, hidden_size, id2vocab)
    print("HS SKIP-GRAM STANDBY")
elif model_name == "nolosshscbow":
    model = NO_LOSS_HS_CBOW(vocab_size, hidden_size, id2vocab)
    print("NO_LOSS_HS_CBOW STANDBY")
elif model_name == "nolosshsskipgram":
    model = NO_LOSS_HS_Skipgram(vocab_size, hidden_size, id2vocab)
    print("NO_LOSS_HS_Skipgram STANDBY")
else :
    print("There is no model " + args.model)
    raise Exception("NO MODEL \n please check your model name")
    
print(f"this model has {hidden_size} hidden size")

print_point = args.print_point

loss_list = []
st_time = time.time()
lrlr_schedule = 0
for epoch in range(max_epoch):
    file_number = 0
    for train_file_path in path_list:
        print(f"learning rate:{lr}")
        data = []
        with open(train_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                line = utiles.normalize_text(line)
                data += line.split()
        
        print(f"{'-'*25}finish loading data : {file_number}{'-'*25}")
        
        kkk = word2id.keys()
        word_list = []
        for word in data:
            if word in kkk:
                word_list.append(word2id[word])
            else:
                word_list.append(-1)
                
        context = []
        target = []
        for i in range(len(word_list)):
            window_size = np.random.randint(1,6)
            if i - window_size <= 0: continue
            if i + window_size > len(word_list): continue
            target.append(word_list[i])
            context.append(word_list[i-window_size:i] + word_list[i+1:i+window_size+1])
        
        print(f"{'-'*20}finish create training data{'-'*20}")
        
        if model_name in ['hscbow', 'hsskipgram']:
            loss = 0
            max_iter = len(target)
            
            for sequence in range(max_iter):
                if target[sequence] == -1: continue
                if sum(context[sequence]) <= 0 : continue
                
                loss += model.update(context[sequence],target[sequence], lr=lr)
                
                if sequence % print_point == 0:
                    loss /= print_point
                    cur_time = time.time() - st_time
                    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(cur_time))
                    print(f"epoch :{epoch}/{max_epoch}, file_num :{file_number}/{file_num}, time:{cur_time}, iter : {sequence}/{max_iter}, loss : {loss}")
                    loss_list.append(loss)
                    loss = 0
        
        else :
            max_iter = len(target)
            
            for sequence in range(max_iter):
                if target[sequence] == -1: continue
                if sum(context[sequence]) <= 0 : continue
                
                model.update(context[sequence],target[sequence], lr=lr)
                
                if sequence % print_point == 0:
                    cur_time = time.time() - st_time
                    cur_time = time.strftime("%Hh %Mm %Ss",time.gmtime(cur_time))
                    print(f"epoch :{epoch}/{max_epoch}, file_num :{file_number}/{file_num}, time:{cur_time}, iter : {sequence}/{max_iter}")
                
        file_number += 1
        
        lrlr_schedule += 1
        lr_schedule = 1-((lrlr_schedule)/(max_epoch*file_num))
        if lr_schedule < 0.0001:
            lr = start_lr*0.0001
        else:
            lr = start_lr*lr_schedule

    end_time = time.time() - st_time
    dmp_dic = {}
    dmp_dic['vec'] = model.word_vec
    dmp_dic['loss'] = loss_list
    dmp_dic['time'] = end_time
    if file_num > 100 : 
        data_size = 1600
    else : 
        data_size = 8*file_num
    with open(f"{model_name}_{data_size}M_{hidden_size}{epoch}epoch.pkl","wb") as file:
        pickle.dump(dmp_dic, file)