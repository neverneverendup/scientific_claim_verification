CUDA_VISIBLE_DEVICES=0 nohup python -u rationale_selection_paragraph_encoding_prediction.py \
--repfile roberta-large \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
--dropout 0 \
--bert_dim 1024 \
--sent_repr_dim 1024 \
--k 0 \
--train_batch_size 2 \
--eval_batch_size 25 \
--max_seq_length 512 \
--rationale_selection_model_type arg \
--checkpoint ./model/arg_model_k0_use_ref/pytorch_model.bin \
--output_dir ./model/arg_model_k0_use_ref > predict-k0-use-arg-k0-model.txt 2>&1 &
#--output_dir ./model/author_model_k0_use_ref > author_model_k0_use_ref_predict.txt 2>&1 &
