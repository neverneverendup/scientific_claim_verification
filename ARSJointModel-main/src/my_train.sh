/home/LAB/r-chaowenhan/anaconda3/envs/scifact/bin/python my_main.py
#CUDA_VISIBLE_DEVICES=0 nohup python -u my_main.py \
#--model dmis-lab/biobert-large-cased-v1.1-mnli \
#--batch_size_gpu 8 \
#--epoch 2 \
#--output_dir ./model/biobert_joint_model > biobert_joint_arg_model.txt 2>&1 &
