import numpy as np

FLOAT_SIZE = 4

def load_text(file):

    #print "Load word2vec file {}\n".format(file)
    with open(file, 'rt') as f:
        header = f.readline()
        vocab_size, we_dim = map(int, header.split())
        train_size = len(train.factors[train.FORMS].words)
        we = np.random.uniform(-0.25, 0.25, [train_size, we_dim])           
        #shape = [int(x) for x in meta]
        #print 'shape of w2v matrix: {}x{}' % (vocab_size, we_dim)
        #print 'shape of we matrix: {}x{}' % (train_size, we_dim)
        
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
    print "Load word2vec file %s\n" % file
    

    with open(file, "rb") as f:
        
        # Read header [vocab_size, we_dim]
        header = ""
        while True:
            c = f.read(1)
            #print(c)
            if c == "\n":
                break
            #c = f.read(1).decode('utf-8')
            #c = f.read(1)
            header += c
            
        #header = f.readline()
        vocab_size, we_dim = [int(x) for x in  header.split()]
        train_size = len(train.factors[train.FORMS].words)
        bin_len = FLOAT_SIZE * we_dim
        print 'shape of w2v matrix: %dx%d' % (vocab_size, we_dim)
        print 'shape of we matrix: %dx%d' % (train_size, we_dim)
                
        we = np.random.uniform(-0.25, 0.25, [train_size, we_dim])        
        #word_to_index = {}
        #binary_len = np.dtype('float32').itemsize * we_dim
        
        # Read words
        while n_vecs < vocab_size:
            
            word = ""        
            while True:
                c = f.read(1)
                #print(c)
                if c == " ":
                    break
                word += c
            #print word
            print word    
            
               
            idx = train.factors[train.FORMS].words_map.get(word)
            
            if idx is not None: # 0 returned by get() if not in vocab
                #we[idx] = [float(v) for v in line[1:]] # get vec
                #n_words_added += 1
                binary_vector = f.read(bin_len)
                vec = [ struct.unpack_from('f', binary_vector, i)[0]
                                          for i in range(0, len(binary_vector), FLOAT_SIZE) ]
                    #print vectors[word]                                                                                                                   
                    #print(vectors[word])
                print word
                print vec
                we[idx] = np.array(vec, np.float32)
                
                
            #print(word, idx)
            #if idx != 0: # 0 returned by get() if not in vocab
                #we[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                #print(len(we))
                #word_to_index[word] = idx
            else:
                f.read(bin_len) # skip   
            
            n_vecs += 1
            sys.stdout.write("%d%%\r" % (n_vecs / vocab_size * 100))
            sys.stdout.flush()      
            
        return we #, word_to_index (optional)
    
    #sess.run(cnn.W.assign(initW))    

if __name__ == "__main__":
    
    import numpy as np
    #import tensorflow as tf
    #from tensorflow.contrib import learn
    import morpho_dataset
    import sys
    from os.path import expanduser
    
    #train = morpho_dataset.MorphoDataset("/home/liefe/data/cs/czech-pdt-train.txt", lowercase=True)
    
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
    train_file = sys.argv[2]
    home = expanduser('~')
    train_file = home + '/data/cs/' + train_file 
    train = morpho_dataset.MorphoDataset(train_file, lowercase=False)
    #train = morpho_dataset.MorphoDataset(train_file, lowercase=True)
            

    # Save file in numpy format
    #with open(model_file, 'wb') as f:
        #file = '/home/liefe/py/wv_data/word2vec_cs64.bin'
        
    #model = load_text(model_file) # read text file  
    model = load_bin(model_file) # read text file

    print 'model shape: '
    print model.shape
    #print(index_to_word[14])
    #print('done emebedding..testing')
    idx = train.factors[train.FORMS].words_map.get('odkazy')
    print 'odkazy'
    print idx
    print model[idx,:]
    #print'odkazy: {}, we={}'.format(idx, model[idx,:])) 
    print 'saving model'
    np.save(model_file + '_embedded', model)
      