import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Tuple
import transformers
from transformers import BertTokenizerFast, BertModel
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr, ttest_1samp, wilcoxon
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from utils import *
from plotnine import *
import spacy
import patchworklib as pw
import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')

dat = "MecoL2_11.csv" # No Estonian
raw = pd.read_csv(dat)

# MinMax scale eye tracking measures

print("Preparing inputs")
scaler = MinMaxScaler()
raw["dur"] = scaler.fit_transform(raw.dur.values.reshape(-1,1)) 
scaler = MinMaxScaler()
raw["firstrun.dur"] = scaler.fit_transform(np.array(raw["firstrun.dur"]).reshape(-1,1))

# prepare sentences and texts
sents = prepare_sents(raw)
texts = prepare_texts(raw, nlp)

# add special tokens 
data = []
for i in range(1, 13):
    text = ['[CLS]']
    text.extend(
        [token for sent in sents if sent.text_id == i for token in sent.sent + ['[SEP]']]
    )
    data.append(text)
    
# Bert model and tokenizer config
print("Loading model")
model_name = "google-bert/bert-base-uncased"

tokenizer = BertTokenizerFast.from_pretrained(model_name)

model = BertModel.from_pretrained(model_name,
                                  output_attentions = True)

class Tokenizer_args(BaseModel):
    is_split_into_words: bool = Field(default=True)
    max_seq_len: int = Field(default=256) # enough to hold the sub-tokens as well
    add_special_tokens: bool = Field(default=False) # manually added
    padding: bool = Field(default='max_length')
    return_attention_mask: bool = Field(default=True)
    truncation: Union[str, bool] = Field(default=True)

# tokenize the input
tokenizer_args = Tokenizer_args()
input_ids, input_mask, word_ids = tokenize(tokenizer, data, tokenizer_args)

# get the embedding layer and embed the inputs
word_embedding_layer = model.embeddings.word_embeddings
emb = word_embedding_layer(input_ids)

# pool to word level
pooled_emb, pooled_mask = pooling_fn(batch = emb, word_ids=word_ids)

# pass the embeddings to the model
print("Forward passing")
device = "cuda" if torch.cuda.is_available() else "cpu"

model, pooled_emb, pooled_mask = model.to(device), pooled_emb.to(device), pooled_mask.to(device)
model.eval()
outputs = model(inputs_embeds = pooled_emb,
                attention_mask = pooled_mask)

first_layer_scores = outputs.attentions[0].cpu().detach()

# average over all attention heads
avg_first_layer_scores = torch.mean(first_layer_scores, dim=1)

# remove special tokens 
mask = torch.ones(avg_first_layer_scores.shape)
for idx in range(len(data)):
    text = data[idx]
    for token_id, token in enumerate(text):
        if token in ["[CLS]", "[SEP]"]:
            mask[idx, 0, token_id] = -1
            
first_scores = avg_first_layer_scores * mask

scores = [first_scores[i, 0][first_scores[i, 0]>=0][:texts[i].text_len] for i in range(len(texts))]

text_ids = []
lang_list = []
spearmanrs = []
spearmanps = []
word_pos = []

for text_id, text in enumerate(texts):    

    xs = []
    local_rs = []
    local_ps = []
    ids = []
    langs = []
    for lang in np.unique(text.lang):
        
        aggregated_text = np.zeros(text.text_len)
        model = scores[text_id]
        
        reader_list = [reader for idx, reader in enumerate(text.reader_list) if text.lang[idx] == lang]
        
        skipped = 0
        for idx, reader in enumerate(reader_list):
            human = text.fix_list[idx]
        
            if set(human) == {0}:
                skipped +=1
                continue
            
            aggregated_text = aggregated_text + human
        
        aggregated_text = aggregated_text/(len(reader_list)-skipped)

        try:  
            for begin in range(0, len(aggregated_text), 10): #window size = 10
                end = begin+10
                x = (begin+end)/2
                spearman = spearmanr(aggregated_text[begin:end], model[begin:end])
                
                ids.append(text_id+1)
                xs.append(x)
                local_rs.append(spearman[0])
                local_ps.append(spearman[1])
                langs.append(lang)
        except(Exception):
            print(aggregated_text)
            print(model)
    
    text_ids.extend(ids)
    word_pos.extend(xs)
    lang_list.extend(langs)
    spearmanrs.extend(local_rs)
    spearmanps.extend(local_ps)
    
df = pd.DataFrame({"Position": word_pos, "Spearman's":spearmanrs, "p":spearmanps, "text":text_ids, "lang":lang_list})

p=(
    ggplot(
        data=df,
        mapping=aes(x="Position", y = "Spearman's", color="lang"))+
        geom_line()+
        labs(title=f"Rolling Correlation Along the Texts")+
        facet_wrap("text", nrow = 3)+
        theme_classic()
  )
p.save(path="vis", filename = "plot_rolling.png", width = 10, height = 5, verbose = False, dpi=300)
p