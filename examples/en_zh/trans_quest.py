import logging
import os
import shutil
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import sklearn
from algo.transformers.evaluation import pearson_corr, spearman_corr
from algo.transformers.run_model import QuestModel
from examples.common.util.draw import draw_scatterplot
from examples.common.util.normalizer import fit, un_fit
from examples.en_zh.transformer_config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, transformer_config, SEED, \
    RESULT_FILE, RESULT_IMAGE

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

# TRAIN_FILE = "data/en-zh/train.enzh.df.short.tsv"
# TEST_FILE = "data/en-zh/dev.enzh.df.short.tsv"

# train = pd.read_csv(TRAIN_FILE, sep='\t', error_bad_lines=False)
# test = pd.read_csv(TEST_FILE, sep='\t', error_bad_lines=False)

with open("data/en-zh/train.pickle", 'rb') as f:  
    train = pickle.loads(f.read())

with open("data/en-zh/test.pickle", 'rb') as f:  
    test = pickle.loads(f.read())

with open("data/en-zh/train_vis.pickle", 'rb') as f:  
    vis1 = pickle.loads(f.read())

with open("data/en-zh/test_vis.pickle", 'rb') as f:  
    vis2 = pickle.loads(f.read())
# train
col_name = train.columns.tolist()
col_name.insert(3,'vis')
train_vis = train.reindex(columns=col_name)
train_vis['vis'] = train_vis['vis'].astype(object)
for idx, v in enumerate(vis1):
	train_vis['vis'][idx] = v
# test
col_name = test.columns.tolist()
col_name.insert(3,'vis')
test_vis = test.reindex(columns=col_name)
test_vis['vis'] = test_vis['vis'].astype(object)
for idx, v in enumerate(vis2):
	test_vis['vis'][idx] = v

train = train_vis[['original', 'translation', 'score', 'vis']]

test = test_vis[['original', 'translation', 'score', 'vis']]

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'score': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'score': 'labels'}).dropna()

# train = fit(train, 'labels')
# test = fit(test, 'labels')
# print(train)
print(train)
if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
        test_preds = np.zeros((len(test), transformer_config["n_fold"]))
        for i in range(transformer_config["n_fold"]):

            if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                shutil.rmtree(transformer_config['output_dir'])

            model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                               args=transformer_config)
            train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED*i)
            model.train_model(train, eval_df=eval_df)
            model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1, use_cuda=torch.cuda.is_available(), args=transformer_config)
            result, model_outputs, wrong_predictions = model.eval_model(test, acc=sklearn.metrics.accuracy_score)
            test_preds[:, i] = model_outputs

        test['predictions'] = test_preds.mean(axis=1)

    else:
        model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)
        train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train, eval_df=eval_df, acc=sklearn.metrics.accuracy_score)
        model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
                           use_cuda=torch.cuda.is_available(), args=transformer_config)
        result, model_outputs, wrong_predictions = model.eval_model(test,acc=sklearn.metrics.accuracy_score)
        test['predictions'] = model_outputs


else:
    model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                       args=transformer_config)
    model.train_model(train)
    result, model_outputs, wrong_predictions = model.eval_model(test,acc=sklearn.metrics.accuracy_score)
    test['predictions'] = model_outputs


# test = un_fit(test, 'labels')
# test = un_fit(test, 'predictions')
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(test, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), MODEL_TYPE + " " + MODEL_NAME)
