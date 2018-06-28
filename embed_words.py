import numpy as np

def load_text(file):

    print("Load word2vec file {}\n".format(file))
    with open(file, 'rt') as f:
        header = f.readline()
        vocab_size, we_dim = map(int, header.split())
        train_size = len(train.factors[train.FORMS].words)
        we = np.random.uniform(-0.25, 0.25, [train_size, we_dim])           
        #shape = [int(x) for x in meta]
        print('shape of w2v matrix: {}x{}'.format(vocab_size, we_dim))
        print('shape of we matrix: {}x{}'.format(train_size, we_dim))
        
        #we = np.zeros(shape, np.float32) # embeddings matrix
        
        #index_to_word = []
        #word_to_index = {}
        n_words_added = 0
        for i, line in enumerate(f):
            #print(line)
            line = line.split()
            word = line[0]
            idx = train.factors[train.FORMS].words_map.get(word)
            #print(word, idx)
            # Check if word in training data and if so add to embedding matrix we
            if idx is not None: # 0 returned by get() if not in vocab
                we[idx] = [float(v) for v in line[1:]] # get vec
                n_words_added += 1
                #print('adding {} with idx {} to we, size of we = {}'.format(word, idx, n_words_added))
                #word_to_index[word] = idx
            #else:
                #f.read(binary_len) # skip                
            #vec = [float(v) for v in line[1:]]
            ##print(word, vec)
            #we[i,:] = vec
            #index_to_word.append(word)
            #word_to_index[word] = i
            #if i == 1000:
                #break
        return we 

def load_bin(file):
   
    # Initial matrix with random uniform
    # load any vectors from the word2vec
    print("Load word2vec file {}\n".format(file))
    with open(file, "rb") as f:
        header = f.readline()
        vocab_size, we_dim = map(int, header.split())
        train_size = len(train.factors[train.FORMS].words)
        we = np.random.uniform(-0.25, 0.25, [train_size, we_dim])        
        #word_to_index = {}
        #binary_len = np.dtype('float32').itemsize * we_dim
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('ascii')
                #print(type(ch.decode('utf-8')))
                print(ch)
                if ch == ' ':
                    word = ''.join(word)
                    print('break')
                    break
                if ch != '\n':
                    word.append(ch)  
                    print('new line')
            print('here')
            #idx = vocab_processor.vocabulary_.get(word)
            idx = train.factors[train.FORMS].words_map.get(word)
            print(word, idx)
            if idx != 0: # 0 returned by get() if not in vocab
                we[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                print(len(we))
                #word_to_index[word] = idx
            else:
                f.read(binary_len) # skip    
                
        return we #, word_to_index (optional)
    
    #sess.run(cnn.W.assign(initW))    

if __name__ == "__main__":
    
    import numpy as np
    import tensorflow as tf
    from tensorflow.contrib import learn
    import morpho_dataset
    import sys
   
    train = morpho_dataset.MorphoDataset("/home/liefe/data/cs/czech-pdt-train.txt", lowercase=True)
    
    #train = morpho_dataset.MorphoDataset("/home/liefe/data/cs/train.txt", lowercase=True)
    #train = morpho_dataset.MorphoDataset("/afs/ms/u/l/liefe/data/cs/train.txt", lowercase=True)
    # To read as text
    #file = 'word2vec_cs.txt'
    #we, index_to_word, word_to_index = get_params(file)
    #print(we)
    #print(index_to_word[14])
    #print(word_to_index['odkazy'])
    
    # Read bin file
    model_file = sys.argv[1]
    # Save file in numpy format
    #with open(model_file, 'wb') as f:
        #file = '/home/liefe/py/wv_data/word2vec_cs64.bin'
        
    model = load_text(model_file) # read text file
    print('model shape: ', model.shape)
    #print(index_to_word[14])
    print('done emebedding..testing')
    idx = train.factors[train.FORMS].words_map.get('odkazy')
    print('odkazy: {}, we={}'.format(idx, model[idx,:])) 
    print('saving model')
    np.save(model_file + '_embedded', model)
        
        
    

    
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
