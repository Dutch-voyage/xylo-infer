for p_attn in 0.60 0.70 0.80 0.90; do
  for layer_budget in 512 1024 2048; do
    for window_size in 32; do
      for steps_between_cache_compressions in 128; do
        python -m eval.test_aime_evict \
            --data_source aime25 \
            --enforce_eager True \
            --compress_method vanilla_topp \
            --if_fake_compress False \
            --layer_budget ${layer_budget} \
            --window_size ${window_size} \
            --steps_between_cache_compressions ${steps_between_cache_compressions} \
            --p_attn ${p_attn} 
      done
    done
  done
done