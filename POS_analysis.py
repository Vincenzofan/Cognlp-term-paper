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

nlp = spacy.load('en_core_web_sm')

dat = "MecoL2_11.csv" # No Estonian
raw = pd.read_csv(dat)

scaler_human = StandardScaler()
scaler_model = StandardScaler()
raw["dur"] = scaler_human.fit_transform(raw.dur.values.reshape(-1,1))

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

# get attention scores from layer 1, 6 and 12
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

scaler_model.fit(torch.hstack(scores).numpy().reshape(-1,1))
pos_of_interest = ["PROPN", "VERB", "NOUN", "PRON", "DET", "PART"]
plots = []
stats_list = []

for pos_type in pos_of_interest:
    human = []
    model = []
#     pos_list = []
    lang_list = []
    for text, score in zip(texts, scores):
        for idx, pos in enumerate(text.pos):
            if pos == pos_type:
                model.append(score[idx])
                for lang in np.unique(text.lang):
                    readers = [text.fix_list[r_idx][idx] 
                               for r_idx, reader in enumerate(text.reader_list)
                               if text.lang[r_idx] == lang]
                    human.append(np.mean(readers))
                    lang_list.append(lang)
    
    
    model = scaler_model.transform(np.array(model).reshape(-1,1)).reshape(-1)
#     print(model)
    stats_dict = {}
    for lang in np.unique(lang_list):
        fix = [human[idx] for idx, l in enumerate(lang_list) if l == lang]
        stats = wilcoxon(fix, model)
        stats_dict[lang] = stats
    stats_list.append(stats_dict)
        
    df = pd.DataFrame({"human": human, "lang": lang_list})
    lang_sig = [lang for lang in np.unique(lang_list) if stats_dict[lang].pvalue <0.05]
    df_sig = df[df["lang"].isin(lang_sig)]
    df_nsig = df[~df["lang"].isin(lang_sig)]
    p = (
        ggplot(data=df,
              mapping=aes(x="lang", y="human", color="lang"))+
        geom_boxplot(df_sig)+
        geom_boxplot(df_nsig, color="grey")+
        geom_hline(yintercept = np.mean(model), color="red")+
        labs(title=pos_type, x="Language", y = "Value")+
        theme(axis_text = element_text(size=15))+
        theme_classic()+
        scale_color_discrete(guide=False)
    )
    plots.append(pw.load_ggplot(p, figsize=(4,2)))

patched = (plots[0]|plots[1]|plots[2])/(plots[3]|plots[4]|plots[5])
patched.savefig("vis/plot_pos.png")
patched