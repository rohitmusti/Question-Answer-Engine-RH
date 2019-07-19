python3 data_gen.py --train_topic_num 10 --dev_topic_num 5 --test_topic_num 5
python3 exp1_transform.py --datasplit all
python3 exp1_setup.py
python3 exp1_train.py --eval_steps 4000 --num_epochs 5 