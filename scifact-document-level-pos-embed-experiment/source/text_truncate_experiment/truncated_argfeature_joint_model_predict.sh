CUDA_VISIBLE_DEVICES=0 python -u truncated_argfeature_joint_paragraph_dynamic_model_predict.py \
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
--epoch 2 \
--max_seq_length 512 \
--update_step 10 \
--loss_ratio 6 \
--repfile roberta-large \
--output_dir ./model/predict_discard_method_obj_Sen_truncated_argfeature_tfidf_joint_paragraph_model_k12 \
--checkpoint /home/LAB/r-chaowenhan/gzy/study/scientific_claim_veirification/sota_analysis_132/scifact-study/scifact-document-level-pos-embed-experiment/source/compare_different_rs_train_data/model/tfidf-torch17_joint_arg_model_k12/16_stance_f1_0.6666666666666667_rationale_f1_0.6925465838509317.model \
#> discard_ONLY_method_Sen_truncated_joint_model_train_log.txt 2>&1 &