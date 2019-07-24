python3 data_gen.py --train_topic_num 45 --dev_topic_num 10 --test_topic_num 9
python3 exp1_transform.py --datasplit all
python3 exp1_setup.py --chunk_size 10
#python3 exp1_train.py --eval_steps 4000 --num_epochs 5 --num_train_chunks 91
