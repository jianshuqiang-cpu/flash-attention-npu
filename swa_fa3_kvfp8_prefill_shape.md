# SWA / FA3 KV-FP8 输入 tensor shape（Step5）

## prefill（TP4PP4，chunk 16384）

| 参数 | chunk0 | chunk3 | dtype |
|---|---|---|---|
| `query` | (16384, 16, 192) | (16384, 16, 192) | bfloat16 |
| `key_cache` | (129, 128, 1, 192) | (133, 128, 1, 192) | float8_e4m3fn |
| `value_cache` | (129, 128, 1, 192) | (133, 128, 1, 192) | float8_e4m3fn |
| `k_descale` | (129, 128, 1) | (133, 128, 1) | float32 |
| `v_descale` | (129, 128, 1) | (133, 128, 1) | float32 |
| `cache_seqlens` | (257,) | (257,) | int32 |
| `page_table` | (1, 8192) | (1, 8192) | int32 |
| `cu_seqlens_q` | (258,) | (258,) | int32 |
| `scheduler_metadata` | (6272,) | (6272,) | uint8 |

## decode（TP4DP4）

| 参数 | shape | dtype |
|---|---|---|
| `query` | (1, 16, 192) | bfloat16 |
| `key_cache` | (6, 128, 1, 192) | float8_e4m3fn |
| `value_cache` | (6, 128, 1, 192) | float8_e4m3fn |
| `k_descale` | (6, 128, 1) | float32 |
| `v_descale` | (6, 128, 1) | float32 |
| `cache_seqlens` | (257,) | int32 |
| `page_table` | (1, 8192) | int32 |
| `cu_seqlens_q` | (258,) | int32 |
| `scheduler_metadata` | (6272,) | uint8 |
