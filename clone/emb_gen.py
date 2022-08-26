from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import pickle
import javalang
from typing import List




class SnippetInferInstance:
    """Subcode Inference Data Instance.
    Represents a single sample in the subcode inference task.

    Attributes:
        snippet (str): Snippet (part of function) string.
        context (str): Context (function) string.
        label (int): Label.
        snippet_embd (torch.Tensor): Snippet embedding.
        context_embd (torch.Tensor): Context embedding.
    """

    def __init__(self, snippet: str, context: str, label: int,context_original_string: str):
        assert label in (0, 1)
        self.snippet: str = snippet
        self.context: str = context
        self.label: int = label
        self.context_original_string = context_original_string
        self.snippet_embd: torch.Tensor = None
        self.context_embd: torch.Tensor = None

    def add_block(self, block):
        self.block = block


def load_model():
    root = 'data/'
    lang = 'java'
    word2vec = Word2Vec.load(root + lang + "/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 5
    BATCH_SIZE = 1
    USE_GPU = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                           USE_GPU, embeddings)
    model.load_state_dict(torch.load('astnn_weights.pth'))
    model.eval()
    if USE_GPU:
        model.cuda()
    return model


def load_CodeBERT_emb():
    data:List[SnippetInferInstance] = pickle.load(
        open('/home/liwei/privacy-attack-code-model/ybr/data/snippet_inference_cls_bertbase_15.pkl', 'rb'))
    print(len(data))

    from utils import get_blocks_v1 as func
    from gensim.models.word2vec import Word2Vec

    root = 'data/'
    lang = 'java'
    word2vec = Word2Vec.load(root + lang + "/train/embedding/node_w2v_128").wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]
    def parse_program(func):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree

    def tree_to_index(node):
        token = node.token
        result = [vocab[token].index if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(r):
        blocks = []
        func(r, blocks)
        tree = []
        for b in blocks:
            btree = tree_to_index(b)
            tree.append(btree)
        return tree

    model = load_model()
    for index,instance in enumerate(data):
        try:
            ast = parse_program(instance.context_original_string)
            block = trans2seq(ast)
        except:
            print(index)
        instance.context_embd = model.encode([block]).cpu()[0].detach().numpy()
        # if index>1000:
        #     break
    with open('./data/victim_astnn_attacker_codebert.pkl', 'wb') as f:
        pickle.dump(data, f)


def main():
    load_CodeBERT_emb()


if __name__ == '__main__':
    main()
