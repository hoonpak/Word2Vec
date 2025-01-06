from collections import Counter
from tqdm import tqdm
import re
import numpy as np

def normalize_text(text): #data preprocessing
    text = text.lower()
    
    text = re.sub("’", "'", text)
    text = re.sub("′", "'", text)
    text = re.sub("''", " ", text)
    text = re.sub("'", " ' ", text)
    
    text = re.sub('“', '\"', text)
    text = re.sub('”', '\"', text)
    text = re.sub('"', ' " ', text)

    text = re.sub("\.", " . ", text)
    text = re.sub("<br \/>", " ", text)
    text = re.sub(", ", " , ", text)
    text = re.sub("\(", " ( ", text)
    text = re.sub("\)", " ) ", text)
    text = re.sub("!", " ! ", text)
    text = re.sub("\?", " ? ", text)
    text = re.sub(";", " ", text)
    text = re.sub(":", " ", text)
    text = re.sub("-", " - ", text)
    text = re.sub("=", " ", text)
    text = re.sub("\*", " ", text)
    text = re.sub("\|", " ", text)
    text = re.sub("«", " ", text)

    text = re.sub("[0-9]", " ", text)

    return text

class vocab: #substitute c lang struct
    def __init__(self, word, count):
        self.word = word
        self.cn = count
        self.code = []
        self.point = []
        self.codelen = None
        self.parent_node = None
        self.binary = 0

def vocabulary(path_list, Threshold=30000):
    counter_ = Counter()
    
    for path in tqdm(path_list,desc="data loading..."):
        words = []
        with open(path, 'r', encoding='UTF8') as file:
            lines = file.readlines()
            for line in lines:
                line = normalize_text(line)
                words += line.split()
        counter_.update(words)
    if Threshold == -1:
        select_words = counter_.most_common()
    else:
        select_words = counter_.most_common(Threshold)
        
    # sort by frequency
    word2id = {word: i for i, (word, count) in enumerate(select_words)}
    id2vocab = {i: vocab(word, count) for i, (word, count) in enumerate(select_words)}
    print("Finish the counting")
    return word2id, id2vocab

class Huffman_Tree:
    def __init__(self, id2vocab):
        self.id2vocab = id2vocab
        self.vocab_size = len(id2vocab)

    def create_tree(self):
        nodes = self.id2vocab.copy()
        for _ in range(self.vocab_size):
            nodes[self.vocab_size + _] = vocab(None, 1e15)
                
        pos1 = self.vocab_size - 1
        pos2 = self.vocab_size
        
        for a in tqdm(range(self.vocab_size),desc="create tree"):
            if pos1 >= 0:
                if nodes[pos1].cn < nodes[pos2].cn:
                    min1i = pos1
                    pos1 -= 1
                else:
                    min1i = pos2
                    pos2 += 1
            else:
                min1i = pos2
                pos2 += 1
            
            if pos1 >= 0:
                if nodes[pos1].cn < nodes[pos2].cn:
                    min2i = pos1
                    pos1 -= 1
                else:
                    min2i = pos2
                    pos2 += 1
            else:
                min2i = pos2
                pos2 += 1
            
            nodes[self.vocab_size + a].cn = nodes[min1i].cn + nodes[min2i].cn
            nodes[min1i].parent_node = self.vocab_size + a
            nodes[min2i].parent_node = self.vocab_size + a
            nodes[min2i].binary = 1

        for a in tqdm(range(self.vocab_size),desc="assign the code,point"):
            b = a
            i = 0
            code = []
            point = []
            while True:
                code.append(nodes[b].binary)
                point.append(b)
                i += 1
                b = nodes[b].parent_node
                if b == self.vocab_size*2-2: break
            
            nodes[a].codelen = i
            # nodes[a].code = list(reversed(code))[:-1]
            nodes[a].code = list(reversed(code))
            nodes[a].point = [self.vocab_size - 2]
            point = [p - self.vocab_size for p in point[1:]]
            # point = [p - self.vocab_size for p in point]
            nodes[a].point += list(reversed(point))
        
        self.id2vocab = {index:nodes[index] for index in range(self.vocab_size)}
        
def test_words(path = "./questions-words.txt"):
    with open(path, 'r', encoding = "UTF8") as f:
        temp = f.readlines()

    semantic_words = []
    syntatic_words = []
    for e in temp:
        t = e[:-1].split(" ")
        if t[1] == "gram1-adjective-to-adverb":
            break
        if t[0] == ":":
            continue
        words = [tt.lower() for tt in t]
        semantic_words.append(words)
    for e in temp[::-1]:
        t = e[:-1].split(" ")
        if t[1] == "gram1-adjective-to-adverb":
            break
        if t[0] == ":":
            continue
        words = [tt.lower() for tt in t]
        syntatic_words.append(words)

    return np.array(semantic_words), np.array(syntatic_words)

def analogy_accuracy(semantic_words, syntatic_words, W_in, word2idx):
    
    def statement(query):
        for word in query:
            if word not in word2idx:
                return True
        return False
    
    sem = semantic_words
    syn = syntatic_words

    score = [0,0]
    ratio = [0,0]
    total_num = [0,0]
    
    W_in /= np.linalg.norm(W_in, axis = 1, keepdims = True)
    
    for i,sort in enumerate([syn, sem]):
        j = 0
        for query in tqdm(sort, desc = "scoring"):
            if statement(query):
                continue
            # 1 - 0 + 2 = 3
            query_vec = W_in[word2idx[query[1]]] - W_in[word2idx[query[0]]] + W_in[word2idx[query[2]]]
            norm = np.sqrt((query_vec * query_vec).sum())
            query_vec /= norm
            query_vec = np.expand_dims(query_vec, 0)
            target = word2idx[query[3]]
            
            j += 1
            
            similarity = np.matmul(query_vec, W_in.T)
            similarity = (-1*similarity).argsort()
            top = 0
            
            while True:
                result = similarity[:,top]
                if result in [word2idx[query[0]],word2idx[query[1]],word2idx[query[2]]]:
                    top += 1
                else: break
                
            if target == result:
                score[i] += 1
                
        total_num[i] = j        
        ratio[i] = score[i]/j
        
    return score, total_num, ratio