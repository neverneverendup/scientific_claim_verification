#!/bin/bash
# This script recreates table 4 of the paper.
#
# Usgae: bash script/pipeline.sh [retrieval] [model] [dataset]
# [retrieval] options: "tf-idf", "neural"
# [model_dir] options: "model/author_model_k12"
# [dataset] options: "dev", "test"
# bash evaluate-pipeline.sh tf-idf dev model/author_model_k12 30
# 要提高效果，para的选择是增加k值，为了检索出更多的正样本


retrieval=$1
dataset=$2
model_dir=$3
k=$4
model_type=$5

##################

CUDA_VISIBLE_DEVICES=1 nohup /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python -u rationale_selection_paragraph_encoding_train.py \
--repfile roberta-large \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
--bert_lr 1e-5 \
--lr 5e-6 \
--dropout 0 \
--bert_dim 1024 \
--sent_repr_dim 1024 \
--rationale_selection_model_type arg \
--k 12 \
--use_ref_data \
--train_batch_size 2 \
--eval_batch_size 10 \
--epoch 20 \
--max_seq_length 512 \
--update_step 10 \
--output_dir ./model/arg_model_k12_use_ref > arg_model_k12_use_ref.txt 2>&1 &


###################

# Run rationale selection
#echo; echo "Selecting rationales."
if [ $retrieval == "tf-idf" ]
then
    /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python -u rationale_selection_paragraph_encoding_prediction.py \
        --repfile roberta-large \
        --corpus_file ../../data/para_scifact/corpus.jsonl \
        --train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
        --test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
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
else
    /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python -u rationale_selection_paragraph_encoding_prediction.py \
        --repfile roberta-large \
        --corpus_file ../../data/para_scifact/corpus.jsonl \
        --train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
        --test_file ../../data/para_scifact/claims_dev_scikgat_retrieved.jsonl \
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
fi

####################

# 这里进行abstract的label预测的时候，使用的是上一个环节输出的所有rationale句子，似乎与论文中作者说的限制输出三条不符。
# Run label prediction, using the selected rationales.
echo; echo "Predicting labels."
/home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python transformer.py \
    --corpus ../../data/para_scifact/corpus.jsonl \
    --dataset ../../data/para_scifact/claims_${dataset}.jsonl \
    --rationale-selection ${model_dir}/rationale_selection.jsonl \
    --model ./model/label_roberta_large_fever_scifact \
    --output ${model_dir}/label_prediction.jsonl

####################
# Merge rationale and label predictions.
echo; echo "Merging predictions."
/home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python merge_predictions.py \
    --rationale-file ${model_dir}/rationale_selection.jsonl \
    --label-file ${model_dir}/label_prediction.jsonl \
    --result-file ${model_dir}/merged_predictions.jsonl

####################
# Evaluate final predictions
echo; echo "Evaluating."
/home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python pipeline.py \
    --gold ../../data/para_scifact/claims_${dataset}.jsonl \
    --corpus ../../data/para_scifact/corpus.jsonl \
    --prediction ${model_dir}/merged_predictions.jsonl \
    --output ${model_dir}/pipeline_result_k_${k}_${dataset}_${retrieval}_${model_type}.txt

