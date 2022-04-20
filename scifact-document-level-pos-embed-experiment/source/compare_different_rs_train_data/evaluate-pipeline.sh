#!/bin/bash
# This script recreates table 4 of the paper.
#
# Usgae: bash script/pipeline.sh [retrieval] [model] [dataset]
# [retrieval] options: "tfidf", "scikgat"
# [dataset] options: "dev", "test"
# [model_dir] options: "model/author_model_k12"
# [k]
# [model_type] options: "arg", "para"
# [label_prediction_model_type] options: "verisci", "simple-att"

# bash evaluate-pipeline.sh tfidf dev model/author_model_k12 30 arg simple-att

# 要提高效果，para的选择是增加k值，为了检索出更多的正样本

retrieval=$1
dataset=$2
model_dir=$3
k=$4
model_type=$5
label_prediction_model_type=$6

# PYTHON_SCRIPT="/home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python"
PYTHON_SCRIPT="python"

###################

# Run rationale selection
#echo; echo "Selecting rationales."
#if [ $retrieval == "tfidf" ]
#then
$PYTHON_SCRIPT -u rationale_selection_paragraph_encoding_prediction.py \
--repfile roberta-large \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_dev_${retrieval}_retrieved.jsonl \
--dropout 0 \
--bert_dim 1024 \
--sent_repr_dim 1024 \
--k ${k} \
--train_batch_size 2 \
--eval_batch_size 25 \
--max_seq_length 512 \
--rationale_selection_model_type ${model_type} \
--checkpoint ${model_dir}/pytorch_model.bin \
--output_dir ${model_dir}
#else
#    /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python -u rationale_selection_paragraph_encoding_prediction.py \
#        --repfile roberta-large \
#        --corpus_file ../../data/para_scifact/corpus.jsonl \
#        --train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
#        --test_file ../../data/para_scifact/claims_dev_${retrieval}_retrieved.jsonl \
#        --dropout 0 \
#        --bert_dim 1024 \
#        --sent_repr_dim 1024 \
#        --k ${k} \
#        --train_batch_size 2 \
#        --eval_batch_size 25 \
#        --max_seq_length 512 \
#        --rationale_selection_model_type ${model_type} \
#        --checkpoint ${model_dir}/pytorch_model.bin \
#        --output_dir ${model_dir}
#fi

####################

# 这里进行abstract的label预测的时候，使用的是上一个环节输出的所有rationale句子，似乎与论文中作者说的限制输出三条不符。
# Run label prediction, using the selected rationales.
echo; echo "Predicting labels."
if [ $label_prediction_model_type == "verisci" ]
then
    $PYTHON_SCRIPT transformer.py \
        --corpus ../../data/para_scifact/corpus.jsonl \
        --dataset ../../data/para_scifact/claims_${dataset}.jsonl \
        --rationale-selection ${model_dir}/rationale_selection.jsonl \
        --model ./model/label_roberta_large_fever_scifact \
        --output ${model_dir}/label_prediction.jsonl
else
    $PYTHON_SCRIPT fever_scifact_stance_paragraph_prediction.py \
        --repfile roberta-large \
        --corpus_file ../../data/para_scifact/corpus.jsonl \
        --test_file ../../data/para_scifact/claims_dev_${retrieval}_retrieved.jsonl \
        --dropout 0 \
        --bert_dim 1024 \
        --k 3 \
        --train_batch_size 2 \
        --eval_batch_size 25 \
        --max_seq_length 512 \
        --checkpoint ../../../../paragraphJointModel/fever_pretrained_para_stance_model/scifact_roberta_stance_paragraph.model \
        --output_dir ${model_dir}
fi

####################
# Merge rationale and label predictions.
echo; echo "Merging predictions."
$PYTHON_SCRIPT merge_predictions.py \
    --rationale-file ${model_dir}/rationale_selection.jsonl \
    --label-file ${model_dir}/label_prediction.jsonl \
    --result-file ${model_dir}/merged_predictions.jsonl

####################
# Evaluate final predictions
echo; echo "Evaluating."
$PYTHON_SCRIPT pipeline.py \
    --gold ../../data/para_scifact/claims_${dataset}.jsonl \
    --corpus ../../data/para_scifact/corpus.jsonl \
    --prediction ${model_dir}/merged_predictions.jsonl \
    --output ${model_dir}/pipeline_result_k_${k}_${dataset}_${retrieval}_${model_type}_${label_prediction_model_type}.txt

