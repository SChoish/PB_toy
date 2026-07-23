# PB toy 결과 — noisy 100k · **@100k / @200k** (2026-07-24 02:55 KST)

- 셀: **`@100k / @200k`** mean success (%). csh·dgx(200k 완료) 병합, 출처 구분 없음.
- `*` on @200k = NT sweep best (`scripts/parse_pb_noisy100k_nt_sweep.py`).
- 러닝커브: `PB_toy_learning_curves_noisy100k_csh.png`
- 이 파일은 `scripts/sync_pb_toy_to_pblogs_csh.sh` 워처가 주기적으로 갱신.
- agents: hiql, tr_hiql, pbg, pbf (trl 제외)

## @100k / @200k

| env/task | hiql | tr_hiql | pbg | pbf |
|---|---:|---:|---:|---:|
| ice/lap_1p | 6.4 / 13.6 | 12.8 / 8.0 | 41.6 / 34.4 | 32.8 / 36.8 |
| ice/lap_2p | 40.8 / 16.8 | 58.4 / 1.6 | 44.0 / 36.0 | 48.0 / 44.0 |
| ice/lap_4p | 14.4 / 8.8 | 5.6 / 0.0 | 5.6 / 16.8 | 1.6 / 0.8 |
| grav/lap_1p | 95.2 / 80.0 | 97.6 / 67.2 | 85.6 / 84.8 | 99.2 / 92.8* |
| grav/lap_2p | 96.8 / 83.2 | 40.0 / 98.4 | 68.0 / 80.8 | 100.0 / 100.0* |
| grav/lap_4p | 61.6 / 43.2 | 80.0 / 68.8 | 82.4 / 76.0 | 93.6 / 100.0* |
| anti_grav/lap_1p | 18.4 / 44.8 | 63.2 / 48.0 | 60.8 / 62.4 | 96.0 / 97.6* |
| anti_grav/lap_2p | 39.2 / 53.6 | 33.6 / 36.8 | 48.8 / 60.8 | 78.4 / 94.4* |
| anti_grav/lap_4p | 37.6 / 31.2 | 67.2 / 52.0 | 44.0 / 42.4 | 97.6 / 100.0* |
| planet/swingby | 20.0 / 20.8 | 48.8 / 59.2 | 60.0 / 60.0 | 59.2 / 60.0 |
| blackhole/swingby | 14.4 / 20.8 | 49.6 / 59.2 | 37.6 / 31.2 | 37.6 / 35.2 |
| car_parking | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 12.0 | 0.0 / 0.0 |

## Agent mean

| budget | hiql | tr_hiql | pbg | pbf |
|---|---:|---:|---:|---:|
| @100k | 37.1 | 46.4 | 48.2 | 62.0 |
| @200k | 34.7 | 41.6 | 49.8 | 63.5 |

## NT sweep best (@200k, applied `*` cells)

| env/task | agent | best % | N | T | coverage |
|---|---|---:|---:|---:|---:|
| grav/lap_1p | pbf | 92.8 | 8 | 1 | 19/19 |
| grav/lap_2p | pbf | 100.0 | 16 | 0.5 | 19/19 |
| grav/lap_4p | pbf | 100.0 | 8 | 0.25 | 10/19 |
| anti_grav/lap_1p | pbf | 97.6 | 16 | 0.5 | 24/19 |
| anti_grav/lap_2p | pbf | 94.4 | 1 | 0.5 | 24/19 |
| anti_grav/lap_4p | pbf | 100.0 | 8 | 0.25 | 20/19 |
