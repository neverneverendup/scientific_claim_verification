CUDA_VISIBLE_DEVICES=0 python -u key_sentence_joint_paragraph_dynamic_model_predict.py \
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
--epoch 64 \
--max_seq_length 512 \
--update_step 10 \
--loss_ratio 6 \
--checkpoint /home/LAB/r-chaowenhan/gzy/study/scientific_claim_veirification/sota_analysis_132/scifact-study/scifact-document-level-pos-embed-experiment/source/keyfact/model/keyfact-argjoint-tfidf-torch17-loss-ratio-6_joint_arg_model_k12/18_stance_f1_0.6768447837150127_rationale_f1_0.691131498470948.model \
--output_dir ./model/keyfact_on_dev_best_argjoint_model
# > keyfact_on_dev_best_argjoint_model.txt 2>&1 &

#--checkpoint ../compare_different_rs_train_data/model/tfidf-torch17_joint_arg_model_k12/16_stance_f1_0.6666666666666667_rationale_f1_0.6925465838509317.model \

