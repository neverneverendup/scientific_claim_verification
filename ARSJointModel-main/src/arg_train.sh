CUDA_VISIBLE_DEVICES=0 nohup /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python my_arg_main.py \
--epoch 35 \
--model dmis-lab/biobert-large-cased-v1.1-mnli \
--output_dir ./model/lstm-arg-sen-att_biobert_raw_author_best_model_pretrained_init_keep_bert_para_arg_feature_biobert_joint_model \
--pre_trained_model ./model/BioBert_large_w.model \
> lstm-arg-sen-att-biobert_author_best_model_pretrained_init_keep_bert_para_arg_feature_joint_arg_model.txt 2>&1 &
# --model dmis-lab/biobert-large-cased-v1.1-mnli \
