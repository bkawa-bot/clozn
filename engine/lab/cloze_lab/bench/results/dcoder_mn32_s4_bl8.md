**dcoder: max_new=32 steps=4 block_len=8**

| cache | forwards | mean cache-hit | new tok | steps/tok | tok/s | token-match | text-match | mean-conf delta |
|---|---|---|---|---|---|---|---|---|
| off (exact) | 12 | 0% | 23 | 0.52 | 40.9 | baseline | n/a | n/a |
| delta(refresh=1) | 12 | 56% | 23 | 0.52 | 53.5 | 100.0% | yes | -0.003 |
