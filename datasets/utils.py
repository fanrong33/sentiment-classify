# encoding: utf-8
""" 数据集工具包，处理文本数据、图像数据等

@version 1.0.3 build 20180410
"""

import os
import codecs
import logging
import re
import random
import numpy as np
import jieba

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

# 关闭jiebalog打印，其中 log_level > 10，即大于 logging.DEBUG 级别即可
jieba.setLogLevel(20)


def get_content(fullname):
    """ 读取文件内容, 并拼接所有段落为一行
    """
    try:
        with codecs.open(fullname, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            lines = [w.strip() for w in lines] # 去掉 \r  \r\n
    except Exception as e:
        return ''
    return ' '.join(lines)

def merge_folder_data(dir):
    """ 将目录原始数据合并到一个txt文件
    neg/...  ->   neg.txt
    pos/...  ->   pos.txt
    """
    dir = os.path.expanduser(dir)
    
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    for foldername in classes:
        logger.info("running "+ foldername +" files.")
        
        output_file = os.path.join(dir, foldername+'.txt') #输出文件
        with codecs.open(output_file, 'w', 'utf8') as output_fp:
            i = 0
            #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
            for parent,dirnames,filenames in os.walk(os.path.join(dir, foldername)):
                for filename in filenames:
                    content = get_content(os.path.join(dir, foldername, filename))
                    if content!= '':
                        output_fp.writelines(content + "\n")
                        i = i+1
                
        logger.info("Merged "+str(i)+" files.")

def clean_character_sentence(string):
    """ 清洗中文文本
    """
    if string != '': 
        string = string.strip()
        # intab = ""
        # outtab = ""
        # trantab = string.maketrans(intab, outtab)
        # pun_num = string.punctuation + string.digits
        # string = string.encode('utf-8')
        # string = string.translate(trantab, pun_num)
        # string = string.decode("utf8")
        # 去除文本中的英文和数字, 去掉数字？
        string = re.sub("[a-zA-Z0-9]", "", string)
        # 去除文本中的中文符号和英文符号
        string = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", string) 
    return string

def segment_character(line, stopwords):
    """ 中文字符jieba分词，并删除停用词。这样则与英文文本按空格分隔保持一致
        stopwords = [w.strip() for w in codecs.open('stopwords.txt', 'r', encoding='utf-8').readlines()]
        Params
            line: '北京天安门'
        Return
            '北京 天安门'
    """
    seg_list = jieba.cut(line, cut_all=False)    
    seg_sentence = []
    for word in seg_list:
        seg_sentence.append(word)

    new_sentence = []
    for word in seg_sentence:
        if word in stopwords:
            continue
        else:
            new_sentence.append(word)

    return ' '.join(new_sentence) 

def prepare_text(source_file, target_file, stopwords):
    """ 准备数据，文本分词（对大文件分别调用前面clean_character_sentence和segment_character处理为与英文文本按空格分隔保持一致）
    在进行分词前，需要对文本进行去除数字、字母和特殊符号的处理
    neg.txt  ->   neg_cut.txt
    pos.txt  ->   pos_cut.txt
    """
    with codecs.open(source_file, 'r', encoding='utf-8') as fp:
        with codecs.open(target_file, 'w', encoding='utf-8') as target_fp:
            line_num = 1
            for line in fp.readlines():
                if line_num % 100 == 0:
                    logger.info('processing %d lines' % line_num)
                line = clean_character_sentence(line)
                seg_line = segment_character(line, stopwords)
                target_fp.writelines(seg_line + '\n')
                line_num = line_num + 1

            logger.info('Processed total %d lines.' % line_num)

def build_words_to_vectors(sentences, max_seq_length, word2vec):
    """ 返回word_list在word_vectors的特征词向量
    Params
        word_list: ['北京', '天安门']
        word2vec: 特指格式为 gensim.models.Word2Vec, eg {'北京':[...]}
    Return
        [[...], [...]]
    """
    input_data = []
    vocab_size = 0

    for line in sentences:
        word_list = line.strip().split(' ')  # 去掉line最后的\n
        vectors = []
        index = 0
        for word in word_list:
            try:
                vectors.append(word2vec[word].tolist())  # 生产环境推荐从 MongoDB 获取词向量，节省内存空间
                if vocab_size == 0:
                    vocab_size = len(word2vec[word])
            except KeyError:
                # TODO 记录word2vectors中的未登录词
                pass
            index = index + 1
            if index >= max_seq_length:
                break
        for _ in range(max_seq_length - len(vectors)):
            zeros_vec = np.zeros((vocab_size), dtype='int32')
            vectors.insert(0, zeros_vec.tolist())  # 前面补0

        input_data.append(vectors)

    return input_data

def build_vectors(filename, max_seq_length, word2vec):
    """ 构建文档的词向量 (按行调用build_words_to_vectors逐行构建)
    注意：这里最后输出为 time_step, input_size 维度
    Params
        filename: 已预先处理为与英文文本按空格分隔保持一致
        word2vec: 特指格式为 gensim.models.Word2Vec, eg {'北京':[...]}
    Return
        [[[...400dim...],
          [...400dim...],
          [...400dim...]]]
    """
    with codecs.open(filename, 'rb', encoding='utf-8') as fp:
        sentences = fp.readlines()
        sentences = [line.strip() for line in sentences]
        sentences = [line for line in sentences if line != '']
        
        input_data = build_words_to_vectors(sentences, max_seq_length, word2vec)
        
    return input_data



def build_ids_matrix(filename, max_seq_length, word_list):
    """ 构建文档的ids matrix矩阵
    Params
        filename: 已预先处理为与英文文本按空格分隔保持一致
        word_list: 单词列表, eg ['北京', '天安门']
    Return
        [[...],  batch_size, time_step
         [...]] 
    """
    input_data = []
    with codecs.open(filename, 'rb', encoding='utf-8') as fp:
        lines = fp.readlines()
        batch_size = len(lines)
        input_data = np.zeros((batch_size, max_seq_length), dtype='int32')
        line_num = 0
        for line in lines:
            ####
            line = clean_sentences(line)

            index = 0
            for word in line.split(' '):
                if line_num < max_seq_length:
                    try:
                        input_data[line_num][index] = word_list.index(word)
                    except ValueError:
                        input_data[line_num][index] = batch_size-1
                index = index + 1
            line_num = line_num + 1
    return input_data

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def clean_english_sentence(string):
    """ 清洗英文文本
    """
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string)  # 去除特殊字符
    return string



def build_token_to_idx(sentences):
    """ 
    对应数据集的词库：存储数据集中的所有出现的不重复的词，且编号。(英文版，以空格分隔)
    """
    token_to_ix = {}
    for sent in sentences:
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix


def build_class_to_idx(classes):
    """ 根据列表创建索引
    params:
        ['dog', 'cat']
    return:
        {'dog': 0, 'cat': 1}
    """
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


def find_classes(dir):
    """ 根据目录返回对应的分类和索引
        - dir/
          |- dog/
          |- cat/
        ['cat', 'dog'], {'cat': 0, 'dog': 1}
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    """ 根据目录和分类索引创建数据集, （第一个元素只是路径，未加载）
    params:
        dir
        class_to_idx
    return:
        [ (path, class), ... ]
    """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                # if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])  # tuple()
                images.append(item)
    return images


def prepare_sequence(seqs, word2idx, max_seq_length):
    """
    返回句子长度为SENTENCE_LENGTH，不够的前面补0
    param:
        seqs: 以空格分割的原始句子列表, 兼容中英文
        word2idx: dict
        max_seq_length: 
    return:
        
    """
    sents_idx = []
    for seq in seqs:
        idxs = [word2idx[w] for w in seq.split(' ')]

        for _ in range(max_seq_length - len(idxs)):
            idxs.insert(0, 0)  # 前面补0, 后面补零,一直没有拟合

        sents_idx.append(idxs)

    return sents_idx


# 推荐在数据集层事先预处理，而不是每次getitem的时候处理
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def prepare_sequence2(seqs, words_list, max_seq_length):
    sents_idx = []
    for sent in seqs:
        cleaned_line = clean_sentences(sent)
        split = cleaned_line.split()

        data = []

        i = 0
        for word in split:
            try:
                data.append(words_list.index(word))
            except ValueError:
                data.append(399999)
            i = i + 1
            if i >= max_seq_length:
                break

        for _ in range(max_seq_length - len(data)):
            # idxs.append(0)  # 后面补零，一直没有拟合?!?
            data.insert(0,0)  # 前面补0      

        sents_idx.append(data)
    return sents_idx


    

