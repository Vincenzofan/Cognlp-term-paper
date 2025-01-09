from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Union, Tuple
import transformers
from transformers import BertTokenizerFast, BertModel
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr, ttest_1samp, wilcoxon
import PIL
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import colors

class Sentence(BaseModel):
    sent_id: int
    text_id: int
    sent: List[str]
    
class Text(BaseModel):
    text_id: int
    text: List[str]
    pos: List[str]
    text_len: int
    reader_list: List[str]
    fix_list: List[List[int|float]]
    gaze_list: List[List[int|float]]
    lang: List[str]|None = None
        
    @field_validator("fix_list", mode="before")
    def check_fix(cls, fix_list):
        # assign a fixation duration of 0 to skipped words
        fix_list = [np.where(np.isnan(np.array(fix)), 0, np.array(fix)) for fix in fix_list]
        return fix_list
    
    @field_validator("gaze_list", mode="before")
    def check_gaze(cls, gaze_list):
        # assign a fixation duration of 0 to skipped words
        gaze_list = [np.where(np.isnan(np.array(fix)), 0, np.array(fix)) for fix in gaze_list]
        return gaze_list
    
    @model_validator(mode="after")
    def check_integraty(cls, self):
        before = len(self.fix_list)
        for idx, fix in enumerate(self.fix_list):
#             complete records
            new_list = [(fix, reader) for _, (fix, reader) 
                        in enumerate(zip(self.fix_list, self.reader_list)) 
                        if len(fix) == self.text_len]
            self.fix_list, self.reader_list = zip(*new_list)
        after = len(self.fix_list)
        if before - after:
            print(f"Droped {before - after} record(s) for incompleteness.")
        print(f"{after} complete records for text {self.text_id}: {self.text[:2]}")
        
        self.lang = [reader[:2] for reader in self.reader_list]
        
        return self

class Tokenizer_args(BaseModel):
    is_split_into_words: bool = Field(default=True)
    max_seq_len: int = Field(default=64) # should be enough to hold the sub-tokens as well
    add_special_tokens: bool = Field(default=False) # manually added
    padding: bool = Field(default='max_length')
    return_attention_mask: bool = Field(default=True)
    truncation: Union[str, bool] = Field(default=True)
        
def tokenize(tokenizer: BertTokenizerFast,
             data: List[List[str]],
             tokenizer_args: Tokenizer_args = Tokenizer_args()):
    
    
    input_ids, input_mask, word_ids = [], [], []
    for sent in data:
    
        encoded = tokenizer.encode_plus(
            sent,
            add_special_tokens=tokenizer_args.add_special_tokens,
            is_split_into_words=tokenizer_args.is_split_into_words,
            max_length=tokenizer_args.max_seq_len,
            padding=tokenizer_args.padding,
            return_attention_mask=tokenizer_args.return_attention_mask,
            truncation=tokenizer_args.truncation    
        )
        input_ids.append(encoded['input_ids'])
        input_mask.append(encoded['attention_mask'])
        word_ids.append(encoded.word_ids())
    
    input_ids = torch.tensor(input_ids)
    input_mask = torch.tensor(input_mask)
    
    # the word ids first have to be converted to np array bc of the Nones
    word_ids = torch.tensor(np.array(word_ids, dtype=np.float32))
    
        
    return input_ids, input_mask, word_ids 

def prepare_sents(df):
    sents = []
    for sentid in range(1, max(df.sentid.values)+1): #sentid starts from 1
        these_sents = df[df.sentid == sentid].sort_values(["uniform_id", "ianum"]).drop_duplicates()
        reader_list = np.unique(these_sents.uniform_id.values)
        text_id = these_sents.itemid.values[0]
        this_sent = these_sents[these_sents.uniform_id == reader_list[0]].ia.values

        sents.append(Sentence(sent_id = sentid,
                              text_id = text_id,
                              sent = this_sent))
    return sents

def prepare_texts(df, nlp):
    texts = []
    for textid in range(1, max(df.itemid.values)+1): #textid starts from 1
        these_texts = df[df.itemid == textid].sort_values(["uniform_id", "ianum"]).drop_duplicates()
        reader_list = np.unique(these_texts.uniform_id.values) #get readers for these texts
        #get one of these texts
        this_text = these_texts[these_texts.uniform_id == reader_list[0]].ia.values 
        
#         pos-tagging
        text_str = " ".join(this_text)
        no_punct = [w for w in nlp(text_str) if not w.is_punct]
        pos = []
        for w in nlp(text_str):
            if w.is_punct:
                continue #punctuations are attached to last words of sentences
            elif w.text in ["’s", "'s"]:
                pos.pop()
                pos.append('CONTRA') #contracted forms 
                continue
            pos.append(w.pos_)
        text_len = len(this_text)
        assert text_len == len(pos)
        total_fix_list = [these_texts[these_texts.uniform_id == reader].dur.values for reader in reader_list]
        gaze_list = [these_texts[these_texts.uniform_id == reader]["firstrun.dur"] for reader in reader_list]

        texts.append(Text(
            text_id = textid,
            text = this_text,
            pos = pos,
            text_len = text_len,
            reader_list = reader_list,
            fix_list = total_fix_list,
            gaze_list = gaze_list,
        ))
        
    return texts

def pooling_fn(
    batch: torch.Tensor,  # shape bsz x seq len x emb dim
    word_ids: List[List[int]],
    pool_method: str = 'avg',  # 'avg', 'sum'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Function that takes as input a batch (not a single instance) and pools the sub-words in 
    each instance to word-level. Either average or sum pooling.
    """
    # create an empty tensor to hold the merged embeddings, shape bsz x 0 x emb dim
    merged_emb = torch.empty(batch.shape[0], 0, batch.shape[2])
    
    max_length = batch.shape[1]  # the sequence length
    
    # iterate through all possible word ids
    for word_idx in range(max_length):
        
        # tensor of shape bsz x 1 x emb dim
        # contains True if the word id is the current word idx (that we are looking at), False otherwise
        # if a word was split into several sub-word tokens, all these sub-word tokens will consist of True
        # so we know which ones to merge together
        word_mask = (word_ids == word_idx).unsqueeze(2).repeat(1, 1, 768)  # bsz x seq len x emb dim 
        
        if pool_method == 'avg':
            # multiply the embeddings with the mask so only the subwords belonging to the same word (ID) remain
                # then average them
            pooled_word_emb = torch.mean(batch * word_mask, dim=1).unsqueeze(1)  # bsz x 1 x emb dim
            
        elif pool_method == 'sum':
            # multiply the embeddings with the mask so only the subwords belonging to the same word (ID) remain
                # then sum them
            pooled_word_emb = torch.sum(batch * word_mask, dim=1).unsqueeze(1)  # bsz x 1 x emb dim
            
        else:
            raise NotImplementedError('this kind of pooling has not been implemented.')
        
        # concatenate the pooled word embedding to the tensor containing all word embeddings for each sequence in the batch
        merged_emb = torch.cat([merged_emb, pooled_word_emb], dim=1)
      #  print(merged_emb.shape)
    
    # create the new attention mask, shape bsz x seq len
    attention_mask = torch.sum(merged_emb, 2).bool().int()
    
    return merged_emb, attention_mask

def run_spearman(human_total, human_gaze, model_first, model_middle, model_last):
    
    spearman_fisrt_total = spearmanr(human_total, model_first)
    spearman_middle_total = spearmanr(human_total, model_middle)
    spearman_last_total = spearmanr(human_total, model_last)

    spearman_fisrt_gaze = spearmanr(human_gaze, model_first)
    spearman_middle_gaze = spearmanr(human_gaze, model_middle)
    spearman_last_gaze = spearmanr(human_gaze, model_last)
    
    rs = [spearman_fisrt_total.statistic, spearman_middle_total.statistic, spearman_last_total.statistic,
         spearman_fisrt_gaze.statistic, spearman_middle_gaze.statistic, spearman_last_gaze.statistic]
    
    ps = [spearman_fisrt_total.pvalue, spearman_middle_total.pvalue, spearman_last_total.pvalue,
         spearman_fisrt_gaze.pvalue, spearman_middle_gaze.pvalue, spearman_last_gaze.pvalue]
    
    measures = ["Total fixation duration", "Total fixation duration", "Total fixation duration",
               "Gaze duration", "Gaze duration", "Gaze duration"]
    
    layers = ["First", "Middle", "Last", "First", "Middle", "Last"]
    
    return rs, ps, measures, layers 
    
def vis(text,
        scores,
        width = 1000,
        height = 700,
        margin = 10,
        padding = 5,
        font_size = 20,
        hspacing = 15,
        vspacing = 20,
        save_path = "vis/example.png"
       ):
    max_score, min_score = max(scores), min(scores)

    # rescale attention scores
    scores_rescaled = [int(255*((score-min_score)/(max_score-min_score))) for score in scores]
    img = Image.new("RGB", (width, height), "white")
    font = ImageFont.truetype("COURIER.TTF",font_size)
    draw = ImageDraw.Draw(img)
    position = (margin+padding, margin+padding)
    line_height = font_size

    for idx, word in enumerate(text):
        score_rescaled = scores_rescaled[idx]

        left, top, right, bottom = draw.textbbox(position, word, font=font)

        if right > width:
            position = (margin+padding, line_height+vspacing+position[1]) #new line
        left, top, right, bottom = draw.textbbox(position, word, font=font)


        if score_rescaled <128:
            fill = (2*score_rescaled, 2*score_rescaled, 255)
        else:
            fill = (255, 512 -2*score_rescaled , 512-2*score_rescaled)
        draw.rectangle((position[0]-padding,
                        position[1]-padding,
                        position[0]+right-left+padding,
                        position[1]+line_height+padding),
                       fill=fill)
        draw.text(position, word, font=font, fill="black")
        position = (right+padding+hspacing, position[1])
    print(scores_rescaled[103])
    norm = colors.Normalize(vmax=max_score, vmin=min_score)
    cmp = plt.get_cmap("bwr", 256)
    image = plt.imshow(img, norm=norm, cmap=cmp)
    plt.axis('off')
    plt.colorbar(image, norm = norm, cmap = cmp,
                 ticks = [min_score, (min_score+max_score)/2, max_score],
                 location = "bottom", orientation = "horizontal",
                 shrink = 0.5,
                 pad = -0.3
                )
    plt.savefig(save_path, dpi=300)