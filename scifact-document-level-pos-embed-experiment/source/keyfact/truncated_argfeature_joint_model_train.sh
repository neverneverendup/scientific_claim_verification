CUDA_VISIBLE_DEVICES=1 nohup python -u truncated_argfeature_joint_paragraph_dynamic_model_train.py \
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
--epoch 22 \
--max_seq_length 512 \
--update_step 10 \
--loss_ratio 6 \
--repfile roberta-large \
--output_dir ./model/discard_method_objective_Sen_truncated_argfeature_tfidf_joint_paragraph_model_k12 \
> discard_method_objective_Sen_truncated_joint_model_train_log.txt 2>&1 &