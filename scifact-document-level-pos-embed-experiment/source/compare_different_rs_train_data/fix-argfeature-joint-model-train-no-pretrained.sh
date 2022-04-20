CUDA_VISIBLE_DEVICES=1 nohup python -u argfeature_joint_paragraph_dynamic_model_train-fix-stance-part.py \
--repfile roberta-large \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/claims_train_biosent_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_dev_biosent_retrieved.jsonl \
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
--stance_part_type arg \
--output_dir ./model/arg_torch17_joint_arg_model_k12 > arg_torch17_joint_arg_model_k12.txt 2>&1 &