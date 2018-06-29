#83;40601;0c!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import  WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments

    if len(sys.argv) < 4:
        #print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp, we_dim = sys.argv[1:4]
    we_dim = int(we_dim)
    # Train model with dim=400
    model = Word2Vec(LineSentence(inp), size=we_dim, window=5, min_count=5, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)

    model.save(outp)
    # model.wv.save_word2vec_format(outp + '.bin', binary=True)
    model.wv.save_word2vec_format(outp + '.txt', binary=False)
    
    # Deprecated
    #outp2 = outp + '_bin'
    #model.save_word2vec_format(outp2, binary=True)
