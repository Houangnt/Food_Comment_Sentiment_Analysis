import pandas as pd
from models import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import torch
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import argparse
from transformers.modeling_utils import * 
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
from utils import * 
from Clean import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--test_path', type=str, default='data/test_translated_cleaned.csv')
parser.add_argument('--dict_path', type=str, default="PhoBERT_base_transformers/dict.txt")
parser.add_argument('--config_path', type=str, default="PhoBERT_base_transformers/config.json")
parser.add_argument('--rdrsegmenter_path', type=str, default="vncorenlp/VnCoreNLP-1.2.jar")
parser.add_argument('--pretrained_path', type=str, default='PhoBERT_base_transformers/model.bin')
parser.add_argument('--max_sequence_length', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ckpt_path', type=str, default='./models')
parser.add_argument('--bpe-codes', default="PhoBERT_base_transformers/bpe.codes",type=str, help='path to fastBPE BPE')

args = parser.parse_args()
bpe = fastBPE(args)
rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m') 

# Load model
config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    num_labels=1
)
model_bert = RobertaForAIViVN.from_pretrained(args.pretrained_path, config=config, attn_implementation="eager")

if torch.cuda.is_available():
    model_bert.cuda()
    map_location = None
else:
    print("CUDA is not available. Using CPU.")
    model_bert.cpu()
    map_location = torch.device('cpu')
# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file("PhoBERT_base_transformers/dict.txt")

if torch.cuda.device_count():
    print(f"Testing using {torch.cuda.device_count()} gpus")
    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta
else:
    tsfm = model_bert.roberta

data_test = pd.read_csv(args.test_path)
test_df = pd.DataFrame({'id': data_test['RevId'], 'text': data_test['Comment']})
test_df = test_df.dropna(how='all')
test_df = test_df.reset_index(drop=True)

# Clean text
test_df['text'] = clean_text_test(test_df['text'].values)

# Tokenization
test_df['text'] = test_df['text'].progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
X_test = convert_lines(test_df, vocab, bpe, args.max_sequence_length)

# Prediction
preds_en = []
try:
    state_dict = torch.load(os.path.join(args.ckpt_path, "model_0.bin"), map_location=torch.device('cpu'))
    missing_keys, unexpected_keys = model_bert.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
except Exception as e:
    print(f"Error loading state_dict: {e}")
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
model_bert.eval()

pbar = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
for i, (x_batch,) in pbar:
    y_pred = model_bert(x_batch, attention_mask=(x_batch > 0))
    y_pred = y_pred.view(-1).detach().cpu().numpy()
    preds_en = np.concatenate([preds_en, y_pred])

preds_en = sigmoid(preds_en)
test_df["Rating"] = preds_en 

# Combine with original comments
my_submission = pd.DataFrame({
    'RevId': test_df["id"],
    'Comment': test_df["text"],
    'Rating': test_df["Rating"]
})

# Save to CSV
my_submission.to_csv('submission.csv', index=False)
