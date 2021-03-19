import argparse
import numpy as np

def read_item_index_to_entity_id_file():
   file = './MKR-data/item_index2entity_id.txt'
   print('reading item index to entity id file: ' + file + ' ...')
   i =0 
   with open(file,'r',encoding='utf-8') as fp:
       for line in fp.readlines():
           data = line.strip().split('\t')
           item_index = data[0]
        #    print(item_index)
           satori =data[1]
           item_index_old2new[item_index] = i
           entity_id2index[satori]= i
           i =i+1

def convert_rating():
    file = './MKR-data/BX-Book-Ratings.csv'

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_rating=dict()
    user_neg_rating =dict()

    with open(file,'r',encoding='utf-8') as fp:
        for line in fp.readlines()[1:]:
            data = line.strip().split(";")
            # 去引号
            data = list(map(lambda x:x[1:-1],data))
            item_index_old = data[1]
            if item_index_old not in item_index_old2new:
                continue
            item_index = item_index_old2new[item_index_old]
            user_index_old =data[0]
            rating = float(data[2])
            if rating>=0:
                if user_index_old not in user_pos_rating:
                    user_pos_rating[user_index_old]=set()
                user_pos_rating[user_index_old].add(item_index)
            else:
                if user_index_old not in user_neg_rating:
                    user_neg_rating[user_index_old]=set()
                user_neg_rating[user_index_old].add(item_index)
    print('converting rating file....')
    rating_file='./MKR-data/ratings_final.txt'
    user_cnt=0
    user_index_old2new = dict()
    with open(rating_file,'w',encoding='utf-8') as fp:
        for user_index_old,pos_item_set in user_pos_rating.items():      
            if user_index_old not in user_index_old2new:
                user_index_old2new[user_index_old]=user_cnt
                user_cnt += 1
            user_index = user_index_old2new[user_index_old]

            for item in pos_item_set:
                fp.write('%d\t%d\t1\n' % (user_index, item))
            unwatch_set = item_set-pos_item_set
            if user_index_old in user_neg_rating:
                unwatch_set -= user_neg_rating[user_index_old]
            for item in np.random.choice(list(unwatch_set),size=len(pos_item_set),replace=False):
                fp.write('%d\t%d\t0\n' % (user_index, item))
        fp.close()
    print('number of user:%d'%user_cnt)
    print('number of item:%d'%len(item_set))



def convert_kg():
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0
    file = './MKR-data/kg.txt'
    writerFile ='./MKR-data/kg_final.txt'
    with open(file,'r',encoding='utf-8') as fp,open(writerFile,'w',encoding='utf-8') as w:
        for line in fp.readlines():
            data = line.strip().split('\t')
            head_old = data[0]
            relation_old =data[1]
            tail_old =data[2]
            if head_old not in entity_id2index:
                continue
            head = entity_id2index[head_old]
            if tail_old not in entity_id2index:
                entity_id2index[tail_old]=entity_cnt
                entity_cnt+=1
            tail =entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old]=relation_cnt
                relation_cnt+=1
            relation = relation_id2index[relation_old]
            w.write('%d\t%d\t%d\n' % (head, relation, tail))
        w.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)







if __name__=='__main__':
    # main函数下为全局变量
    np.random.seed(555)
    entity_id2index=dict()
    relation_id2index=dict()
    item_index_old2new=dict()
    read_item_index_to_entity_id_file()

    convert_rating()
    convert_kg()





