**dream: max_new=48 steps=4 block_len=8 quant=nf4**

| cache | forwards | mean cache-hit | new tok | steps/tok | tok/s | token-match | text-match | mean-conf delta |
|---|---|---|---|---|---|---|---|---|
| off (exact) | 12 | 0% | 23 | 0.52 | 27.1 | baseline | n/a | n/a |
| delta(refresh=1) | 12 | 73% | 23 | 0.52 | 40.3 | 100.0% | yes | -0.000 |
