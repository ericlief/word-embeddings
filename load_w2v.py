import numpy as np

def get_params(file):

    print('Processing file')
    with open(file, 'r+') as f:
        #print(f)
        meta = f.readline().split()
        shape = [int(x) for x in meta]
        print(shape)
        we = np.zeros(shape, np.float32)
        index_to_word = []
        word_to_index = {}
        for i, line in enumerate(f):
            #print(line)
            line = line.split()
            word = line[0]
            vec = [float(v) for v in line[1:]]
            #print(word, vec)
            we[i,:] = vec
            index_to_word.append(word)
            word_to_index[word] = i
            if i == 1000:
                break
        return we, index_to_word, word_to_index


def load(file):
   
    # Initial matrix with random uniform
    # load any vectors from the word2vec
    print("Load word2vec file {}\n".format(file))
    with open(file, "rb") as f:
        header = f.readline()
        vocab_size, we_dim = map(int, header.split())
        we = np.random.uniform(-0.25, 0.25, (len(train.factors[train.FORMS].words), we_dim))        
        #word_to_index = {}
        binary_len = np.dtype('float32').itemsize * we_dim
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            #idx = vocab_processor.vocabulary_.get(word)
            idx = train.factors[train.FORMS].words_map.get(word)
            print(word, idx)
            if idx != 0: # 0 returned by get() if not in vocab
                we[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                word_to_index[word] = idx
            else:
                f.read(binary_len) # skip    
                
        return we #, word_to_index (optional)
    
    #sess.run(cnn.W.assign(initW))    

if __name__ == "__main__":
    
    import numpy as np
    import tensorflow as tf
    from tensorflow.contrib import learn
    import morpho_dataset
    
    train = morpho_dataset.MorphoDataset("/home/liefe/data/cs/test.txt", lowercase=True)
    
    # To read as text
    #file = 'word2vec_cs.txt'
    #we, index_to_word, word_to_index = get_params(file)
    #print(we)
    #print(index_to_word[14])
    #print(word_to_index['odkazy'])
    
    # Read bin file
    with open('wv_we', 'wb') as f:
        file = 'word2vec_cs.bin'
        we = load(file)
        print(we.shape)
        #print(index_to_word[14])
        idx = train.factors[train.FORMS].words_map.get('odkazy')
        print('odkazy: {}, we={}'.format(idx, we[idx,:])) 
        np.save(we, f)
        
        
    

    
    #x_text = ['This is a cat','This must be boy', 'This is a a dog']
    #max_document_length = max([len(x.split(" ")) for x in x_text])
    
    ### Create the vocabularyprocessor object, setting the max lengh of the documents.
    #vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    
    ### Transform the documents using the vocabulary.
    #x = np.array(list(vocab_processor.fit_transform(x_text)))    
    #print(x)
    
    ### Extract word:id mapping from the object.
    #vocab_dict = vocab_processor.vocabulary_._mapping
    #print(vocab_dict)
    
    ### Sort the vocabulary dictionary on the basis of values(id).
    ### Both statements perform same task.
    ##sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
    #sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
    #print(sorted_vocab)
    ### Treat the id's as index into list and create a list of words in the ascending order of id's
    ### word with id i goes at index i of the list.
    #vocabulary = list(list(zip(*sorted_vocab))[0])
    
    #print(vocabulary)
    #print(x)
    #print(vocab_processor.vocabulary_.get('fucks'))    
#model = gensim.models.Word2Vec.load('word2vec_cs')
#words = model.most_similar('muž')
#print(words)
#print(model.similarity('muž', 'žena'))
#print(model.most_similar('žena'))
#print(model.similarity('šlapka', 'žena'))
#print(model.wv['muž'].shape) # -> (400,)
# print(model.wv.shape) # -> KeyedVectors has no shape
