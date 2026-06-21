**dream: max_new=48 steps=8 block_len=8 quant=nf4**

| cache | forwards | mean cache-hit | new tok | steps/tok | tok/s | token-match | text-match | mean-conf delta |
|---|---|---|---|---|---|---|---|---|
| off (exact) | 24 | 0% | 23 | 1.04 | 12.8 | baseline | n/a | n/a |
| delta(refresh=1) | 24 | 78% | 23 | 1.04 | 20.5 | 97.9% | no | +0.008 |
