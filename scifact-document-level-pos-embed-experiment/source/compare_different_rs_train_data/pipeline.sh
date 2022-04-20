#!/bin/bash
#
# This script recreates table 4 of the paper.
#
# Usgae: bash script/pipeline.sh [retrieval] [model] [dataset]
# [retrieval] options: "oracle", "open"
# [model] options: "oracle-rationle", "zero-shot", "verisci"
# [dataset] options: "dev", "test"

retrieval=$1
model=$2
dataset=$3

if [ $model = "zero-shot" ]
then
    rationale_training_dataset="fever"
    rationale_threshold=0.025
    label_training_dataset="fever"
else
    rationale_training_dataset="scifact"
    rationale_threshold=0.5
    label_training_dataset="fever_scifact"
fi

echo "Running pipeline on ${dataset} set."

####################

# Download data.
bash script/download-data.sh

####################

# Download models.
bash script/download-model.sh rationale roberta_large ${rationale_training_dataset}
bash script/download-model.sh label roberta_large ${label_training_dataset}

####################

# Create a prediction folder to store results.
rm -rf prediction
mkdir -p prediction

####################

# Run abstract retrieval.
echo; echo "Retrieving abstracts."
if [ $retrieval == "oracle" ]
then
    python3 verisci/inference/abstract_retrieval/oracle.py \
        --dataset data/claims_${dataset}.jsonl \
        --output prediction/abstract_retrieval.jsonl
else
    python3 verisci/inference/abstract_retrieval/tfidf.py \
        --corpus data/corpus.jsonl \
        --dataset data/claims_${dataset}.jsonl \
        --k 3 \
        --min-gram 1 \
        --max-gram 2 \
        --output prediction/abstract_retrieval.jsonl
fi
# python3 verisci/inference/abstract_retrieval/tfidf.py --corpus scifact_data/data/corpus.jsonl --dataset scifact_data/data/claims_dev.jsonl --k 3 --min-gram 1 --max-gram 2 --output prediction/abstract_retrieval-.jsonl
####################

# Run rationale selection
echo; echo "Selecting rationales."
if [ $model == "oracle-rationale" ]
        # 如果pipeline第一步中检索的摘要位于数据的evidence abstracts在集合中，说明abstract检索对了
        #，把其rationale的所有句子都加入evidence列表中
        # 否则如果检索到的abstract是错的，则evidence列表是空的，后续abstract label prediction环节直接判NEI
then
    python3 verisci/inference/rationale_selection/oracle.py \
        --dataset data/claims_${dataset}.jsonl \
        --abstract-retrieval prediction/abstract_retrieval.jsonl \
        --output prediction/rationale_selection.jsonl
else
    python3 verisci/inference/rationale_selection/transformer.py \
        --corpus data/corpus.jsonl \
        --dataset data/claims_${dataset}.jsonl \
        --threshold ${rationale_threshold} \
        --abstract-retrieval prediction/abstract_retrieval.jsonl \
        --model model/rationale_roberta_large_${rationale_training_dataset}/ \
        --output-flex prediction/rationale_selection.jsonl
fi

####################

# 这里进行abstract的label预测的时候，使用的是上一个环节输出的所有rationale句子，似乎与论文中作者说的限制输出三条不符。
# Run label prediction, using the selected rationales.
echo; echo "Predicting labels."
python3 verisci/inference/label_prediction/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --rationale-selection prediction/rationale_selection.jsonl \
    --model model/label_roberta_large_${label_training_dataset} \
    --output prediction/label_prediction.jsonl

####################
# 这里合并selected rationales和预测的abstract label，去除了预测label为NEI的摘要，保留非NEI摘要和相应的evidence sentence
# {"id": 236, "evidence": {"4388470": {"sentences": [3, 6, 7], "label": "SUPPORT"}}}
# Merge rationale and label predictions.
echo; echo "Merging predictions."
python3 verisci/inference/merge_predictions.py \
    --rationale-file prediction/rationale_selection.jsonl \
    --label-file prediction/label_prediction.jsonl \
    --result-file prediction/merged_predictions.jsonl

####################


# Evaluate final predictions
echo; echo "Evaluating."
python3 verisci/evaluate/pipeline.py \
    --gold data/claims_${dataset}.jsonl \
    --corpus data/corpus.jsonl \
    --prediction prediction/merged_predictions.jsonl

# python verisci/evaluate/pipeline.py --gold scifact_data/data/claims_dev.jsonl --corpus scifact_data/data/corpus.jsonl --prediction prediction/merged_predictions.jsonl

# parahraph那篇文章 rs 和 lp模块还是连着的，只是一起训练了，但是不还是存在误差传播吗
#  注意这里是在dev集上执行的pipeline测试结果，论文中是在test set上的pipeline测试结果，这点要注意
            sentence_selection  sentence_label  abstract_label_only  abstract_rationalized
precision            0.524590        0.468852             0.553073               0.525140
recall               0.437158        0.390710             0.473684               0.449761
f1                   0.476900        0.426230             0.510309               0.484536

# oracle abstract的测试结果
{
  "sentence_selection": {
    "precision": 0.7941176470588235,
    "recall": 0.5901639344262295,
    "f1": 0.677115987460815
  },
  "sentence_label": {
    "precision": 0.7132352941176471,
    "recall": 0.5300546448087432,
    "f1": 0.6081504702194357
  },
  "abstract_label_only": {
    "precision": 0.9096774193548387,
    "recall": 0.6746411483253588,
    "f1": 0.7747252747252746
  },
  "abstract_rationalized": {
    "precision": 0.8516129032258064,
    "recall": 0.631578947368421,
    "f1": 0.7252747252747253
  }
}

# rs环节替换para，各指标大概提高三个点
{
  "sentence_selection": {
    "precision": 0.7667731629392971,
    "recall": 0.6557377049180327,
    "f1": 0.706921944035346
  },
  "sentence_label": {
    "precision": 0.6900958466453674,
    "recall": 0.5901639344262295,
    "f1": 0.6362297496318116
  },
  "abstract_label_only": {
    "precision": 0.9058823529411765,
    "recall": 0.7368421052631579,
    "f1": 0.812664907651715
  },
  "abstract_rationalized": {
    "precision": 0.8411764705882353,
    "recall": 0.6842105263157895,
    "f1": 0.7546174142480211
  }
}

# 分析open设置下 para rs环节的提高，各环节提高大概十个点，就离谱
{
  "sentence_selection": {
    "precision": 0.6036745406824147,
    "recall": 0.6284153005464481,
    "f1": 0.6157965194109772
  },
  "sentence_label": {
    "precision": 0.5433070866141733,
    "recall": 0.5655737704918032,
    "f1": 0.5542168674698795
  },
  "abstract_label_only": {
    "precision": 0.6033057851239669,
    "recall": 0.6985645933014354,
    "f1": 0.647450110864745
  },
  "abstract_rationalized": {
    "precision": 0.5578512396694215,
    "recall": 0.645933014354067,
    "f1": 0.598669623059867
  }
}