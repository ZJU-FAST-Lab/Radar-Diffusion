python scripts/image_sample_radar.py --training_mode edm  --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun  --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 128 --num_channels 64 --num_heads 4 --num_res_blocks 3 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --weight_schedule karras --in_ch 2 --out_ch 1 dataset_dir