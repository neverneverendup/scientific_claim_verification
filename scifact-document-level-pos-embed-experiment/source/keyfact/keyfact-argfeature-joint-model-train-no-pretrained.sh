CUDA_VISIBLE_DEVICES=0 nohup python -u key_sentence_joint_paragraph_dynamic_model_train.py \
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
--repfile roberta-large \
--output_dir ./model/keyfact-argjoint-tfidf-torch17-loss-ratio-6_joint_arg_model_k12 \
> keyfact-argjoint-tfidf-torch17-loss_ratio_6_joint_arg_model_k12.txt 2>&1 &