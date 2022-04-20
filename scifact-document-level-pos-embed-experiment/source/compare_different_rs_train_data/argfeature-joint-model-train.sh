
#/home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python -u argfeature_joint_paragraph_dynamic_model_train.py \
#--repfile roberta-large \
#--corpus_file ../../data/para_scifact/corpus.jsonl \
#--train_file ../../data/para_scifact/claims_train_biosent_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_biosent_retrieved.jsonl \
#--bert_lr 1e-5 \
#--lr 5e-6 \
#--dropout 0 \
#--bert_dim 1024 \
#--sent_repr_dim 1024 \
#--k 3 \
#--train_batch_size 2 \
#--eval_batch_size 2 \
#--epoch 20 \
#--max_seq_length 512 \
#--update_step 10 \
#--pre_trained_model /home/LAB/r-chaowenhan/gzy/study/fever_roberta_joint_paragraph_dynamic_1_149999.model \
#--output_dir ./model/fever_initialized_joint_arg_model_k12

CUDA_VISIBLE_DEVICES=1 nohup python -u argfeature_joint_paragraph_dynamic_model_train.py \
--repfile roberta-large \
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
--epoch 35 \
--max_seq_length 512 \
--update_step 10 \
--pre_trained_model ./model/scifact_roberta_joint_paragraph_dynamic_fine_tune_ratio=6_lr=5e-6_bert_lr=1e-5_FEVER=5_scifact=12_downsample_good.model \
--output_dir ./model/only-bert-keep-tfidf-xiangci-pretrained-word-att-joint_arg_model_k12 > only-bert-reserve-word-att-joint_arg_model_k12.txt 2>&1 &

#--pre_trained_model /home/LAB/r-chaowenhan/gzy/study/fever_roberta_joint_paragraph_dynamic_1_149999.model \


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
#--pre_trained_model /home/LAB/r-chaowenhan/gzy/study/fever_roberta_joint_paragraph_dynamic_1_149999.model \
#--output_dir ./model/torch17_fever_initialized_joint_arg_model_k12 > torch17_fever_initialized_joint_arg_model_k12.txt 2>&1 &