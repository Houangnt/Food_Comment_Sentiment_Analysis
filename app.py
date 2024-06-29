from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
from torch import nn
from vncorenlp import VnCoreNLP
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import torch.utils.data
import argparse
from transformers import RobertaConfig
from models import RobertaForAIViVN
from utils import *
from Clean import *
import os
import numpy as np

app = Flask(__name__)

args = {
    'test_path': 'data/test_translated_cleaned.csv',
    'dict_path': "PhoBERT_base_transformers/dict.txt",
    'config_path': "PhoBERT_base_transformers/config.json",
    'rdrsegmenter_path': "vncorenlp/VnCoreNLP-1.2.jar",
    'pretrained_path': 'PhoBERT_base_transformers/model.bin',
    'max_sequence_length': 250,
    'batch_size': 64,
    'ckpt_path': './models',
    'bpe_codes': "PhoBERT_base_transformers/bpe.codes"
}

args = argparse.Namespace(**args)

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
else:
    print("CUDA is not available. Using CPU.")
    model_bert.cpu()

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file(args.dict_path)

# Load checkpoint
try:
    state_dict = torch.load(os.path.join(args.ckpt_path, "model_0.bin"), map_location=torch.device('cpu'))
    missing_keys, unexpected_keys = model_bert.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
except Exception as e:
    print(f"Error loading state_dict: {e}")

if torch.cuda.device_count():
    print(f"Testing using {torch.cuda.device_count()} gpus")
    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta
else:
    tsfm = model_bert.roberta

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    
    # Clean and preprocess text
    cleaned_text = clean_text_test([text])[0]  # Assuming clean_text_test returns a list
    tokenized_text = ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(cleaned_text)])
    X_test = convert_lines(pd.DataFrame({'text': [tokenized_text]}), vocab, bpe, args.max_sequence_length)
    
    # Prediction
    with torch.no_grad():
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        model_bert.eval()
        
        preds_en = []
        for x_batch in test_loader:
            y_pred = model_bert(x_batch[0], attention_mask=(x_batch[0] > 0))
            y_pred = y_pred.view(-1).detach().cpu().numpy()
            preds_en = np.concatenate([preds_en, y_pred])
        
        rating = float(preds_en[0])  # Assuming single prediction
    
    return jsonify({'rating': rating})

if __name__ == '__main__':
    app.run(debug=True)
