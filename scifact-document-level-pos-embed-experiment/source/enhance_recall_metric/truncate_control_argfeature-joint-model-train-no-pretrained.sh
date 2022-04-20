
# 基于biobert训练
#CUDA_VISIBLE_DEVICES=1 nohup /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python -u argfeature_joint_paragraph_dynamic_model_train.py \
#--corpus_file ../../data/para_scifact/corpus.jsonl \
#--train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
#--bert_lr 1e-5 \
#--lr 5e-6 \
#--dropout 0 \
#--bert_dim 1024 \
#--sent_repr_dim 1024 \
#--k 12 \
#--train_batch_size 1 \
#--eval_batch_size 20 \
#--epoch 25 \
#--max_seq_length 512 \
#--update_step 10 \
#--abstract_lambda 4.36745 \
#--rationale_lambda 11.78625 \
#--repfile dmis-lab/biobert-large-cased-v1.1-mnli \
#--output_dir ./model/optuna-biobert-abslam-4.36745-rationlam-11.78625-tfidf-torch17-joint_arg_model_k12 \
#> optuna-biobert-abslam-4.36745-rationlam-11.78625-torch17-joint_arg_model_k12.txt 2>&1 &

# dmis-lab/biobert-large-cased-v1.1-mnli \

# roberta-large \

# 基于roberta训练
CUDA_VISIBLE_DEVICES=0 /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python -u truncate_control_with_arg_argfeature_joint_paragraph_dynamic_model_train.py \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
--bert_lr 1e-5 \
--lr 5e-6 \
--dropout 0 \
--bert_dim 1024 \
--sent_repr_dim 1024 \
--k 12 \
--train_batch_size 1 \
--eval_batch_size 20 \
--epoch 25 \
--max_seq_length 512 \
--update_step 10 \
--abstract_lambda 1 \
--rationale_lambda 6 \
--repfile roberta-large \
--output_dir ./model/truncate-control-roberta-abslam-1-rationlam-6-tfidf-torch17-joint_arg_model_k12 \
#> truncate-control-roberta-abslam-1-rationlam-6-torch17-joint_arg_model_k12.txt 2>&1 &