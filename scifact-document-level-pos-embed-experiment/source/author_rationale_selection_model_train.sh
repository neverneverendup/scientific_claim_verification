#CUDA_VISIBLE_DIVICES=1 python -u author_scifact_rationale_paragraph_train.py --repfile roberta-large --corpus_file ../data/para_scifact/corpus.jsonl --train_file ../data/para_scifact/claims_train_tfidf_retrieved.jsonl --test_file ../data/para_scifact/claims_dev_tfidf_retrieved.jsonl --bert_lr 1e-5 --lr 5e-6 --dropout 0 --bert_dim 1024 --batch_size 2 --epoch 20 --max_seq_length 512 --update_step 10 --k 3 --output_dir ../model/author_paragraph_encoding_rationale_selection_model-original-parameters > author_scifact_rationale_paragraph_train-log.txt 2>&1 &
CUDA_VISIBLE_DIVICES=0 python author_scifact_rationale_paragraph_train.py --repfile roberta-large --corpus_file ../data/para_scifact/corpus.jsonl --train_file ../data/para_scifact/claims_train_tfidf_retrieved.jsonl --test_file ../data/para_scifact/claims_dev_tfidf_retrieved.jsonl --bert_lr 1e-5 --lr 5e-6 --dropout 0 --bert_dim 1024 --train_batch_size 2 --eval_batch_size 8 --epoch 20 --max_seq_length 512 --update_step 10 --k 0 --output_dir ../model/no-ref-data-author_paragraph_encoding_rationale_selection_model-original-parameters
