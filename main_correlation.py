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

dat = "data/MecoL2_11.csv" # No Estonian
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

# get attention scores from layer 1, 6 and 12
first_layer_scores = outputs.attentions[0].cpu().detach()
middle_layer_scores = outputs.attentions[5].cpu().detach()
last_layer_scores = outputs.attentions[11].cpu().detach()

# average over all attention heads
avg_first_layer_scores = torch.mean(first_layer_scores, dim=1)
avg_middle_layer_scores = torch.mean(middle_layer_scores, dim=1)
avg_last_layer_scores = torch.mean(last_layer_scores, dim=1)

# remove special tokens 
mask = torch.ones(avg_first_layer_scores.shape)
for idx in range(len(data)):
    text = data[idx]
    for token_id, token in enumerate(text):
        if token in ["[CLS]", "[SEP]"]:
            mask[idx, 0, token_id] = -1
            
first_scores = avg_first_layer_scores * mask
middle_scores = avg_middle_layer_scores * mask
last_scores = avg_last_layer_scores * mask

first_scores = [first_scores[i, 0][first_scores[i, 0]>=0][:texts[i].text_len] for i in range(len(texts))]
middle_scores = [middle_scores[i, 0][middle_scores[i, 0]>=0][:texts[i].text_len] for i in range(len(texts))]
last_scores = [last_scores[i, 0][last_scores[i, 0]>=0][:texts[i].text_len] for i in range(len(texts))]

# correlation analysis 
print("Running correlations")
# no aggregation
text_ids = []
reader_ids = []
langs = []
spearmanrs = []
spearmanps = []
measures = []
layers = []

for text_id, text in enumerate(tqdm(texts)):
    for idx, reader in enumerate(text.reader_list):
        human_total = text.fix_list[idx]
        human_gaze = text.gaze_list[idx]
        
        model_first = first_scores[text_id]
        model_middle = middle_scores[text_id]
        model_last = last_scores[text_id]
        
        if set(human_total) == {0}:
            continue # skipped entirely
        
        if set(human_gaze) == {0}:
            continue
            
        rs, ps, ms, ls = run_spearman(human_total, human_gaze, model_first, model_middle, model_last)
        
        for i in range(6):
            text_ids.append(text_id+1)
            reader_ids.append(reader)
            langs.append(text.lang[idx])
        
        spearmanrs.extend(rs)
        spearmanps.extend(ps)
        measures.extend(ms)
        layers.extend(ls)
        
results = pd.DataFrame({"text_id": text_ids, 
                        "reader_id": reader_ids,
                        "lang": langs,
                        "spearman_r": spearmanrs,
                        "spearman_p": spearmanps,
                        "measures": measures,
                        "layers": layers
                       })

# save results 
results.to_csv("results/no_aggregation_correlation.csv")

# average over participants 
text_ids = []
reader_ids = []
langs = []
spearmanrs = []
spearmanps = []
measures = []
layers = []

for text_id, text in enumerate(tqdm(texts)):
    
    for lang in np.unique(text.lang):
        
#         holders 
        aggregated_total = np.zeros(text.text_len)
        aggregated_gaze = np.zeros(text.text_len)
        
        model_first = first_scores[text_id]
        model_middle = middle_scores[text_id]
        model_last = last_scores[text_id]
        
#         get readers of this language group
        reader_list = [reader for idx, reader in enumerate(text.reader_list) if text.lang[idx] == lang]
    
        skipped = 0
        for idx, reader in enumerate(reader_list):
            human_total = text.fix_list[idx]
            human_gaze = text.gaze_list[idx]
        
            if set(human_total) == {0}:
                continue # skipped entirely

            if set(human_gaze) == {0}:
                continue
            
            aggregated_total = aggregated_total + human_total
            aggregated_gaze = aggregated_gaze + human_gaze
        
        aggregated_total = aggregated_total/(len(reader_list)-skipped)
        aggregated_gaze = aggregated_gaze/(len(reader_list)-skipped)

        rs, ps, ms, ls = run_spearman(aggregated_total, aggregated_gaze,
                                      model_first, model_middle, model_last)
        
        for i in range(6):
            text_ids.append(text_id+1)
            reader_ids.append(reader)
            langs.append(lang)
        
        spearmanrs.extend(rs)
        spearmanps.extend(ps)
        measures.extend(ms)
        layers.extend(ls)

avg_results = pd.DataFrame({"text_id": text_ids, 
                        "reader_id": reader_ids,
                        "lang": langs,
                        "spearman_r": spearmanrs,
                        "spearman_p": spearmanps,
                        "measure": measures,
                        "layer": layers
                       })

# save results 
avg_results.to_csv("results/avg_correlation.csv")        

# visualization
avg_results["layer"] = avg_results["layer"].astype('category').cat.reorder_categories(["First", "Middle", "Last"])

p = (
    ggplot(data=avg_results,
      mapping=aes(x="lang", y="spearman_r", fill="lang"))+
    geom_boxplot(width=0.15,
                 outlier_shape = "",
                 position=position_nudge(x=0.30))+
    geom_violin(width=0.5,
                style="right",
                position=position_nudge(x=0.45))+
    geom_point(data = avg_results[avg_results["spearman_p"]<0.05],
               size=1.2,
               stroke = 0.2,
               position=position_jitter(width=0.1))+
    geom_point(data = avg_results[avg_results["spearman_p"]>0.05],
               size=1.2,
               alpha=0.3,
               stroke = 0.2,
               position=position_jitter(width=0.1))+
#     coord_flip()+
#     scale_x_discrete(expand=(0, 0.6))+
    scale_color_discrete(guide=False)+
    facet_grid("measure", "layer")+
    labs(title = "Spearman Correlation Between Model and Human Attention",
        x = "Language", y = "Spearman's r")+
    theme_classic()
)

p.save(path="vis", filename = "plot_correlation.png", width = 10, height = 5, verbose = False, dpi=300)
p
