for layer_budget in 4096; do
  for window_size in 32; do
    for steps_between_cache_compressions in 128; do

      python -m eval.test_aime_evict \
          --compress_method rkv \
          --layer_budget ${layer_budget} \
          --window_size ${window_size} \
          --steps_between_cache_compressions ${steps_between_cache_compressions}

      python -m eval.test_aime_evict \
          --compress_method snapkv \
          --layer_budget ${layer_budget} \
          --window_size ${window_size} \
          --steps_between_cache_compressions ${steps_between_cache_compressions}
    done
  done
done