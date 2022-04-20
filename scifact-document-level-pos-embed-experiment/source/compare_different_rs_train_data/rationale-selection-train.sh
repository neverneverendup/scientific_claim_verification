#CUDA_VISIBLE_DEVICES=0 nohup python -u rationale_selection_paragraph_encoding_train.py \
#--repfile roberta-large \
#--corpus_file ../../data/para_scifact/corpus.jsonl \
#--train_file ../../data/para_scifact/claims_train_with_dev_tfidf_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
#--bert_lr 1e-5 \
#--lr 5e-6 \
#--dropout 0 \
#--bert_dim 1024 \
#--sent_repr_dim 1024 \
#--rationale_selection_model_type arg \
#--k 12 \
#--use_ref_data \
#--train_batch_size 2 \
#--eval_batch_size 10 \
#--epoch 20 \
#--max_seq_length 512 \
#--update_step 10 \
#--output_dir ./model/train-with-dev-tfidf-arg_model_k12_use_ref > train-with-dev-tfidf-arg_model_k12_use_ref.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u rationale_selection_paragraph_encoding_train.py \
--repfile dmis-lab/biobert-large-cased-v1.1-mnli \
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
--output_dir ./model/mnli-biobert-tfidf-arg_model_k12_use_ref > mnli-biobert-tfidf-arg_model_k12_use_ref.txt 2>&1 &

#dmis-lab/biobert-large-cased-v1.1-mnli