for layer_budget in 1024 2048 4096; do
  for window_size in 32; do
    for steps_between_cache_compressions in 128; do

      python -m eval.test_aime_baseline \
          --data_source umathtop50 \
          --enforce_eager True \
          --compress_method snapkv \
          --layer_budget ${layer_budget} \
          --window_size ${window_size} \
          --steps_between_cache_compressions ${steps_between_cache_compressions}
    done
  done
done