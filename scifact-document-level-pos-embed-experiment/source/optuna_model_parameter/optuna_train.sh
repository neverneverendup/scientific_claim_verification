# 基于biobert训练
CUDA_VISIBLE_DEVICES=0 nohup /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python -u optuna_argfeature_joint_paragraph_dynamic_model_train.py \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/optuna_sample_claims_train_tfidf_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
--bert_lr 1e-5 \
--lr 5e-6 \
--dropout 0 \
--bert_dim 1024 \
--sent_repr_dim 1024 \
--k 12 \
--train_batch_size 1 \
--eval_batch_size 32 \
--epoch 20 \
--max_seq_length 512 \
--update_step 10 \
--loss_ratio 6 \
--repfile dmis-lab/biobert-large-cased-v1.1-mnli \
--output_dir ./model/optuna-biobert-tfidf-torch17-joint_arg_model_k12 \
> optuna-biobert-tfidf-torch17_joint_arg_model_k12.txt 2>&1 &

# dmis-lab/biobert-large-cased-v1.1-mnli