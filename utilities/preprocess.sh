source ~/anaconda3/etc/profile.d/conda.sh

conda activate transformers_4_15_0
python preprocessing/preprocess.py res/train/KNOW_2017.csv res/test/KNOW_2017_test.csv --data_type=2017 --use_bert_embeddings
python preprocessing/preprocess.py res/train/KNOW_2018.csv res/test/KNOW_2018_test.csv --data_type=2018 --use_bert_embeddings
python preprocessing/preprocess.py res/train/KNOW_2019.csv res/test/KNOW_2019_test.csv --data_type=2019 --use_bert_embeddings
python preprocessing/preprocess.py res/train/KNOW_2020.csv res/test/KNOW_2020_test.csv --data_type=2020 --use_bert_embeddings

conda activate transformers_2_8_0
python preprocessing/preprocess.py res/train/KNOW_2017.csv res/test/KNOW_2017_test.csv --data_type=2017 --use_simcse_embeddings
python preprocessing/preprocess.py res/train/KNOW_2018.csv res/test/KNOW_2018_test.csv --data_type=2018 --use_simcse_embeddings
python preprocessing/preprocess.py res/train/KNOW_2019.csv res/test/KNOW_2019_test.csv --data_type=2019 --use_simcse_embeddings
python preprocessing/preprocess.py res/train/KNOW_2020.csv res/test/KNOW_2020_test.csv --data_type=2020 --use_simcse_embeddings

conda activate transformers_4_15_0
python preprocessing/merge_outputs.py *.pkl