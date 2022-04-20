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
#--loss_ratio 3 \
#--output_dir ./model/loss_ratio3_torch17_joint_arg_model_k12 > loss_ratio3_torch17_joint_arg_model_k12.txt 2>&1 &

#--train_file ../../data/para_scifact/claims_train_biosent_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_biosent_retrieved.jsonl \

#CUDA_VISIBLE_DEVICES=2 nohup python -u argfeature_joint_paragraph_dynamic_model_train.py \
#--corpus_file ../../data/para_scifact/corpus.jsonl \
#--train_file ../../data/para_scifact/claims_train_with_dev_tfidf_retrieved.jsonl \
#--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
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
#--loss_ratio 6 \
#--repfile roberta-large \
#--output_dir ./model/train-with-dev-tfidf-torch17-loss-ratio-6_joint_arg_model_k12 > train-with-dev-tfidf-torch17-loss_ratio_6_joint_arg_model_k12.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python -u argfeature_joint_paragraph_dynamic_model_train.py \
--corpus_file ../../data/para_scifact/corpus.jsonl \
--train_file ../../data/para_scifact/claims_train_tfidf_retrieved.jsonl \
--test_file ../../data/para_scifact/claims_dev_tfidf_retrieved.jsonl \
--bert_lr 1e-5 \
--lr 5e-6 \
--dropout 0 \
--bert_dim 1024 \
--sent_repr_dim 512 \
--k 12 \
--train_batch_size 1 \
--eval_batch_size 10 \
--epoch 20 \
--max_seq_length 512 \
--update_step 10 \
--loss_ratio 6 \
--repfile roberta-large \
--output_dir ./model/no_ssamplingz_retrain_stance_sentpre_512_roberta_joint_arg_model_k12 \
--pre_trained_model /home/LAB/r-chaowenhan/gzy/study/scientific_claim_veirification/sota_analysis_132/scifact-study/scifact-document-level-pos-embed-experiment/source/compare_different_rs_train_data/model/sentpre_512_roberta_joint_arg_model_k12/14_stance_f1_0.6424870466321243_rationale_f1_0.7105666156202144.model \
> no_ssampling_retrain_stance_sentpre_512_joint_arg_model_k12.txt 2>&1 &