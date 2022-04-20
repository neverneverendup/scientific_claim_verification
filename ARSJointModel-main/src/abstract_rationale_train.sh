CUDA_VISIBLE_DEVICES=0 nohup /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python abstract_rationale_main.py \
--epoch 40 \
--model dmis-lab/biobert-large-cased-v1.1-mnli \
--output_dir ./model/best_lambda_abstract_rationale_biobert_joint_model > best_lambda_abstract_rationale_biobert_joint_model.txt 2>&1 &