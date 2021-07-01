import os
import dill
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator


def buildloaders(store_dict = "./data", batch_size = 64, save = False, use_cache_fields = True):
    
    if use_cache_fields:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading cached fields and Preparing Data for Device: ", device)
        article,summary = loadSavedFields()
        train_data, valid_data, test_data = TabularDataset.splits(path=store_dict, train='train.csv', validation='val.csv', test='test.csv', format='csv', fields=[("text",article),('headline',summary)])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Preparing Data for Device: ", device)

        article = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True, tokenizer_language='en_core_web_trf', include_lengths=True)
        summary = Field(tokenize='spacy',init_token='<sos>',eos_token='<eos>',lower=True, tokenizer_language='en_core_web_trf')

        train_data, valid_data, test_data = TabularDataset.splits(path=store_dict, train='train.csv', validation='val.csv', test='test.csv', format='csv', fields=[("text",article),('headline',summary)])

        article.build_vocab(train_data, min_freq=2)
        summary.build_vocab(train_data, min_freq=2)

        if save:
            os.mkdir("./DumpedField")
            with open("./DumpedField/article.Field","wb")as f:
                dill.dump(article,f)
            with open("./DumpedField/summary.Field","wb")as f:
                dill.dump(summary,f)

    
    train_loader, valid_loader, test_loader = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size, sort_within_batch=True, sort_key = lambda x:len(x.text), device=device)

    return article, summary, train_loader, valid_loader, test_loader

def loadSavedFields():
    articlefile = open("./DumpedField/article.Field","rb+")
    summaryfile = open("./DumpedField/summary.Field","rb+")
    article = dill.load(articlefile)
    summary = dill.load(summaryfile)
    return article,summary