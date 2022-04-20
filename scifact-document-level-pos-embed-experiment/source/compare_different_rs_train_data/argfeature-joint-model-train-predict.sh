#CUDA_VISIBLE_DEVICES=0 nohup python -u argfeature_joint_paragraph_dynamic_model_predict.py \
#--repfile roberta-large \
#--corpus_file ../../data/para_scifact/corpus.jsonl \
#--train_file ../../data/para_scifact/claims_train_biosent_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_scikgat_retrieved.jsonl \
#--bert_lr 1e-5 \
#--lr 5e-6 \
#--dropout 0 \
#--bert_dim 1024 \
#--sent_repr_dim 1024 \
#--k 3 \
#--train_batch_size 1 \
#--eval_batch_size 10 \
#--epoch 20 \
#--max_seq_length 512 \
#--update_step 10 \
#--loss_ratio 3 \
#--checkpoint ./model/torch17_joint_arg_model_k12/13_stance_f1_0.649616368286445_rationale_f1_0.6935724962630793.model \
#--output_dir ./model/torch17_joint_arg_model_k12 > torch17_joint_arg_model_k3_scikgat_predict.txt 2>&1 &
#

CUDA_VISIBLE_DEVICES=0 nohup python -u argfeature_joint_paragraph_dynamic_model_predict.py \
--repfile roberta-large \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_test_tfidf_retrieved.jsonl \
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
--checkpoint ./model/train-with-dev-tfidf-torch17-loss-ratio-6_joint_arg_model_k12/13_stance_f1_1.0_rationale_f1_1.0.model \
--output_dir ./model/train-with-dev-tfidf-torch17-loss-ratio-6_joint_arg_model_k12 > 13_torch17_joint_arg_model_test_predict.txt 2>&1 &


