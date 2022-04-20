CUDA_VISIBLE_DEVICES=1 nohup /home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python my_main.py \
--epoch 40 \
--output_dir ./model/best_lambda_biobert_joint_model > best_lambda_biobert_joint_model.txt 2>&1 &