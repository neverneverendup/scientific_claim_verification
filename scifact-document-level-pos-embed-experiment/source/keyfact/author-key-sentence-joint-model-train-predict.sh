CUDA_VISIBLE_DEVICES=0 python -u author_key_sentence_joint_paragraph_dynamic_model_predict.py \
--repfile roberta-large \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
--bert_lr 1e-5 \
--lr 5e-6 \
--dropout 0 \
--bert_dim 1024 \
--sent_repr_dim 1024 \
--k 30 \
--train_batch_size 1 \
--eval_batch_size 20 \
--epoch 20 \
--max_seq_length 512 \
--update_step 10 \
--loss_ratio 6 \
--checkpoint ./model/scifact_roberta_joint_paragraph_dynamic_fine_tune_ratio=6_lr=5e-6_bert_lr=1e-5_FEVER=5_scifact=12_downsample_good.model \
--output_dir ./model/author_keyfact_on_dev_best_argjoint_model
# > keyfact_on_dev_best_argjoint_model.txt 2>&1 &

