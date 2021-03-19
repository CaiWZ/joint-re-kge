CUDA_VISIBLE_DEVICES=1 python run_knowledgable_recommendation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -rec_test_files valid.dat:test.dat -kg_test_files valid.dat:test.dat -l2_lambda 0 -model_type cke -has_visualization -dataset dbbook2014 -batch_size 400 -embedding_size 100 -learning_rate 0.001 -topn 10 -seed 3 -eval_interval_steps 19520 -training_steps 1952000 -early_stopping_steps_to_wait 97600 -optimizer_type Adam -joint_ratio 0.7 -noshare_embeddings -L1_flag -norm_lambda 1 -kg_lambda 1 -use_st_gumbel -load_ckpt_file tuned_dbbook2014/dbbook2014-bprmf-1540692224.ckpt:tuned_dbbook2014/dbbook2014-transr-1540701160.ckpt -visualization_port 8098

