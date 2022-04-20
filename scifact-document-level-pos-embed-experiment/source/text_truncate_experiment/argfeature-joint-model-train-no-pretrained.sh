#CUDA_VISIBLE_DEVICES=0 nohup python -u argfeature_joint_paragraph_dynamic_model_train.py \
#--repfile roberta-large \
#--corpus_file ../../data/para_scifact/corpus.jsonl \
#--train_file ../../data/para_scifact/claims_train_biosent_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_biosent_retrieved.jsonl \
#--bert_lr 1e-5 \
#--lr 5e-6 \
#--dropout 0 \
#--bert_dim 1024 \
#--sent_repr_dim 1024 \
#--k 12 \
#--train_batch_size 1 \
#--eval_batch_size 10 \
#--epoch 20 \
#--max_seq_length 512 \
#--update_step 10 \
#--loss_ratio 3 \
#--output_dir ./model/loss_ratio3_torch17_joint_arg_model_k12 > loss_ratio3_torch17_joint_arg_model_k12.txt 2>&1 &

#--train_file ../../data/para_scifact/claims_train_biosent_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_biosent_retrieved.jsonl \

#CUDA_VISIBLE_DEVICES=2 nohup python -u argfeature_joint_paragraph_dynamic_model_train.py \
#--corpus_file ../../data/para_scifact/corpus.jsonl \
#--train_file ../../data/para_scifact/claims_train_with_dev_tfidf_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
#--bert_lr 1e-5 \
#--lr 5e-6 \
#--dropout 0 \
#--bert_dim 1024 \
#--sent_repr_dim 1024 \
#--k 12 \
#--train_batch_size 1 \
#--eval_batch_size 10 \
#--epoch 20 \
#--max_seq_length 512 \
#--update_step 10 \
#--loss_ratio 6 \
#--repfile roberta-large \
#--output_dir ./model/train-with-dev-tfidf-torch17-loss-ratio-6_joint_arg_model_k12 > train-with-dev-tfidf-torch17-loss_ratio_6_joint_arg_model_k12.txt 2>&1 &

# 基于biobert训练
CUDA_VISIBLE_DEVICES=1 nohup python -u argfeature_joint_paragraph_dynamic_model_train.py \
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
--eval_batch_size 10 \
--epoch 20 \
--max_seq_length 512 \
--update_step 10 \
--loss_ratio 6 \
--repfile dmis-lab/biobert-large-cased-v1.1-mnli \
--output_dir ./model/mul_claimpararepr-biobert-tfidf-torch17-loss-ratio-6_joint_arg_model_k12 > mul_claimpararepr-biobert--tfidf-torch17-loss_ratio_6_joint_arg_model_k12.txt 2>&1 &

# dmis-lab/biobert-large-cased-v1.1-mnli \