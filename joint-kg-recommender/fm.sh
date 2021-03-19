CUDA_VISIBLE_DEVICES=1 python run_item_recommendation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -rec_test_files valid.dat:test.dat -l2_lambda 1e-5 -negtive_samples 1 -model_type fm -has_visualization -dataset dbbook2014 -batch_size 1024 -embedding_size 100 -learning_rate 0.1 -topn 10 -seed 3 -eval_interval_steps 500 -training_steps 50000 -early_stopping_steps_to_wait 2500 -optimizer_type Adagrad -visualization_port 8097
