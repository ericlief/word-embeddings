import tensorflow as tf
import numpy as	np
import os
import argparse
#import gensim
from gensim.models import Word2Vec
from tensorflow.contrib.tensorboard.plugins import projector


def project():
    # project part of vocab, 10K of 300 dimension
    w2v_10K = np.zeros((10000,400))
    if not os.path.exists("projector"): os.mkdir("projector")
    with open("projector/prefix_metadata.tsv", 'w+') as file_metadata:
        for i,word in enumerate(model.wv.index2word[:10000]):
            w2v_10K[i] = model[word]
            #print(i, word, w2v_10K[i])
            
            #print(type(word.encode('utf-8')))
            #print(type(b'\n'))
            file_metadata.write(str(word.encode('utf-8') + b'\n'))
            
    
    # define the model without training
    sess = tf.InteractiveSession()
    
    with tf.device("/cpu:0"):
        embedding = tf.Variable(w2v_10K, trainable=False, name='prefix_embedding')
    
    tf.global_variables_initializer().run()
    
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./projector', sess.graph)
    
    # adding into projector
    config = projector.ProjectorConfig()
    embed= config.embeddings.add()
    embed.tensor_name = 'fs_embedding:0'
    # project part of vocab, 10K of 300 dimension
    w2v_10K = np.zeros((10000,400))
    if not os.path.exists("projector"): os.mkdir("projector")
    with open("projector/prefix_metadata.tsv", 'w+') as file_metadata:
        for i,word in enumerate(model.wv.index2word[:10000]):
            w2v_10K[i] = model[word]
            #print(i, word, w2v_10K[i])
            
            #print(type(word.encode('utf-8')))
            #print(type(b'\n'))
            file_metadata.write(str(word.encode('utf-8') + b'\n'))
            
    
    # define the model without training
    sess = tf.InteractiveSession()
    
    with tf.device("/cpu:0"):
        embedding = tf.Variable(w2v_10K, trainable=False, name='prefix_embedding')
    
    tf.global_variables_initializer().run()
    
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./projector', sess.graph)
    
    # adding into projector
    config = projector.ProjectorConfig()
    embed= config.embeddings.add()
    #embed.tensor_name = 'fs_embedding:0'
    embed.tensor_name = 'prefix_embedding'
    embed.metadata_path = './projector/prefix_metadata.tsv'
    
    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    
    saver.save(sess, './projector/prefix_model.ckpt', global_step=10000)
    
    embed.metadata_path = './projector/prefix_metadata.tsv'
    
    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    
    saver.save(sess, './projector/prefix_model.ckpt', global_step=10000)
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--input', help='Single embedding file')
    #parser.add_argument('--output', help='Output basename without extension')
    #args = parser.parse_args()
    
    #embeddings_file = args.output + '.npy'
    #vocabulary_file = args.output + '_vocab.txt'
    words = []
    vectors = []
    
    #model = Word2Vec.load_word2vec_format('word2vec_cs', binary=True)
    #model.save_word2vec_format('word2vec_cs.txt', binary=False)	


#with open(args.input, 'rb') as f:
    #for i, line in enumerate(f):
        ##print(i,line)
        #if i==0:
            #continue        
        #fields = line.split()
        #for field in line:
            ##print(field.decode('utf-8'))
            #print(field, sep=' ')
        ##print(line)
        #vector=byte(vector)
        #print(vector.decode('utf-8'))
    
        #print(i)
        #fields = line.split()
        #print(fields)
        #word = fields[0].decode('utf-8')
        #print(word)
        #word = fields[2].encode('utf-8')
        #print(word)
        
        #vector = np.fromiter((float(field.decode('utf-8')) for field in fields[1:]),
                             #dtype=np.float)
        #vector = np.fromiter((x for field in map(lambda x: decode(x), fields[1:])),
                             #dtype=np.float)        
        #words.append(word)
        #vectors.append(vector)



# Projector not working
#project()


def load(): # Load whole gensim wv model
    #model = Word2Vec.load('word2vec_cs')
    model = Word2Vec.load_word2vec_format('word2vec_cs_bin', binary=True)
    model.save_word2vec_format('word2vec_cs.txt', binary=False)	
    #model = Word2Vec.load_word2vec_format('word2vec_cs_bin', binary=False)
    wv = model.wv.syn0
    vocab = model.wv.vocab
    index_to_word = model.wv.index2word
    # This loads only the wv matrix --> (?, dim)
    #we = np.load('word2vec_cs.wv.syn0.npy')
    #print(we.shape) # -> (667123, 400)Word2VecWord2Vec
    #print(we[0])
    
    #print(we.wv.syn0.shape)
    
    #print(len(we.wv.vocab)) # -> (667123, 400)
    print(vocab['na'])
    print(index_to_word[0])
    
    #we = we.wv.syn0
    #print(we[0,:])
    
    #matrix = np.array(vectors)
    #np.save(embeddings_file, matrix)
    #text = '\n'.join(words)

#load()


print('opening file')
with open('word2vec_cs.txt', 'r+') as f:
    #print(f)
    for line in f:
        print(line)
    
    
    #f.write(text.encode('utf-8'))