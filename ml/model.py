from gensim.models import Word2Vec
import pickle
from os import path
from annoy import AnnoyIndex
import numpy as np
import itertools
from copy import deepcopy
import pandas as pd

DATA_STORAGE_PROG_PATH = path.join("data/data_storage_norm_prog.pkl")
data_storage_norm_prog = pickle.load(open(DATA_STORAGE_PROG_PATH, "rb"))

NUM_TREES = 15
VEC_SIZE_EMB = 100
counter = 0
map_id_2_prod_hash = {}
index_title_emb = AnnoyIndex(VEC_SIZE_EMB)
print("Build annoy base")
for prod_hash in data_storage_norm_prog:
    #print(prod_hash)
    title_vec = data_storage_norm_prog[prod_hash] # Вытаскиваем вектор
    
    index_title_emb.add_item(counter, title_vec) # Кладем в анной
    
    map_id_2_prod_hash[counter] = prod_hash # сохраняем мапу - (id в анное -> продукт_id)
    
    counter += 1
    if counter % 10000 == 1:
        print("computed for %d" % counter)
        
index_title_emb.build(NUM_TREES)
print("builded")

DATA_STORAGE_QUEST_PATH = path.join("data/data_storage_norm_quest.pkl")
data_storage_norm_quest = pickle.load(open(DATA_STORAGE_QUEST_PATH, "rb"))

NUM_TREES = 15
VEC_SIZE_EMB = 100
counter = 0
map_id_2_prod_hash_quest = {}
index_title_emb_quest = AnnoyIndex(VEC_SIZE_EMB)
print("Build annoy base")
for prod_hash in data_storage_norm_quest:
    #print(prod_hash)
    title_vec = data_storage_norm_quest[prod_hash] # Вытаскиваем вектор
    
    index_title_emb_quest.add_item(counter, title_vec) # Кладем в анной
    
    map_id_2_prod_hash_quest[counter] = prod_hash # сохраняем мапу - (id в анное -> продукт_id)
    
    counter += 1
    if counter % 10000 == 1:
        print("computed for %d" % counter)
        
index_title_emb_quest.build(NUM_TREES)

df = pd.read_csv("data/programming.csv")
prog_dict = {}
for title,answer in zip(df["Title"], df["Answer"]):
    prog_dict[title] = answer

q_df = pd.read_csv("data/questions.csv")
quest_dict = {}
for title,answer in zip(df["Title"], df["Answer"]):
    quest_dict[title] = answer


W2V_PROG_PATH = path.join("data/word2vec_prog.model")
W2V_QUEST_PATH = path.join("data/word2vec_quest.model")

PROG_DICT_PATH = path.join("data/prog_dict.pkl")
QUEST_DICT_PATH = path.join("data/quest_dict.pkl")

TFIDF_PROG_PATH = path.join("data/tfidf_prog.model")
TFIDF_QUEST_PATH = path.join("data/tfidf_quest.model")

PRODUCT_W2V_PATH = path.join("data/product.model")

ANNOY_PROG_PATH = path.join("data/annoy_prog.model")
#ANNOY_QUEST_PATH = path.join("data/annoy_prog.model")

BINARY_CLASS_PATH = path.join("data/binary.model")

#DATA_STORAGE_QUEST_PATH = path.join("data/data_storage_norm_quest.pkl")

#prog_dict = pickle.load(open("data/prog.txt", "rb"))

class Model:

    def __init__(self):
        self.w2v_prog = Word2Vec.load(W2V_PROG_PATH)
        self.w2v_quest = Word2Vec.load(W2V_QUEST_PATH)

        # self.prog_annoy = AnnoyIndex(100)
        # self.prog_annoy.load(ANNOY_PROG_PATH)

        # self.quest_annoy = AnnoyIndex(100)
        # self.quest_annoy.load(ANNOY_QUEST_PATH)

        self.binary_class = pickle.load(open(BINARY_CLASS_PATH,"rb"))

        
        #self.data_storage_quest_prog = pickle.load(open(DATA_STORAGE_QUEST_PATH), "rb")

    def predict_group(self, question):
        return self.binary_class.predict(question)

    def get_answer(self, question):
        if self.predict_group([question]) == "programming":
            return self.get_prog_answer(question)
        else:
            return self.get_quest_answer(question)
            
    def get_prog_answer(self, question):
        
        

        
        mean = []
        
        cnt = 0
        string = question[0]
        for item in string.split(" "):
            if item in self.w2v_prog.wv.vocab.keys():
                cnt += 1
                if mean == []:
                    mean = deepcopy(self.w2v_prog.wv[item])
                else:
                    mean += self.w2v_prog.wv[item]

        mean /= cnt
            
        



        #print("res: ", res.toarray())
        annoy_res = list(index_title_emb.get_nns_by_vector(mean, 13, include_distances=True))

        print('\n\nСоседи:')

        for annoy_id, annoy_sim in itertools.islice(zip(*annoy_res), 13):
            image_id = map_id_2_prod_hash[annoy_id]
            #print(prog_dict.keys()[0])
            try:
                return prog_dict[df["Title"][image_id]]
            except KeyError:
                return "I don't understand you"
            
        
    
    def get_quest_answer(self, question):
        mean = []
        cnt = 0
        string = question[0]
        for item in string.split(" "):
            if item in self.w2v_quest.wv.vocab.keys():
                cnt += 1
                if mean == []:
                    mean = deepcopy(self.w2v_quest.wv[item])
                else:
                    mean += self.w2v_quest.wv[item]
        if mean == []:
            return "I don't understand you"
        mean /= cnt
            
        



        #print("res: ", res.toarray())
        annoy_res = list(index_title_emb_quest.get_nns_by_vector(mean, 13, include_distances=True))

        print('\n\nСоседи:')

        for annoy_id, annoy_sim in itertools.islice(zip(*annoy_res), 13):
            image_id = map_id_2_prod_hash_quest[annoy_id]
            #print(prog_dict.keys()[0])
            try:
                return quest_dict[df["Title"][image_id]]
            except KeyError:
                return "I don't understand you"
    