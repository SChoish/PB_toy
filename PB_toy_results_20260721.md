# PB toy 결과 값 정리 (2026-07-22 00:47 KST)

- 값 = mean success (%), train-time final eval (`step_10000`).
- `*` = 1k NT sweep **best 값**으로 대체.
- `[w1]` / `[w0]` = distance-weight power. `[w1→0@4k]`처럼 표시된 run은 중간에 설정 변경.
- lap task는 `lap_1p`~`lap_8p`를 자동 인식. 현재 4p/8p는 10k train 결과만 존재.
- mean은 각 dataset에서 현재 완료된 셀만 산술평균하며 `—`는 제외. 1k의 `*`는 NT best로 반영.

## Weight 현황

| dataset | algo | weight별 완료 run |
|---|---|---|
| 1k | tr_hiql | w1: 24 |
| 1k | pbg | w1: 24 |
| 1k | pbf | w1: 24 |
| 10k | tr_hiql | w0: 11, w1: 16 |
| 10k | pbg | w0: 10, w1: 14, w1→0@4k: 1 |
| 10k | pbf | w0: 10, w1: 14, w1→0@6k: 1 |

## NT sweep best (1k, 적용된 셀)

| env/task | algo | policy | NT best | cell (N,T) | coverage | weight |
|---|---|---|---:|---|---:|---|
| anti_grav/lap_2p | pbf | expert | 60.0 | N1 T1 | 24/24 | w1 |
| anti_grav/lap_2p | pbf | noisy | 2.4 | N2 T1 | 24/24 | w1 |
| anti_grav/lap_2p | pbf | random | 0.0 | N1 T1 | 24/24 | w1 |
| anti_grav/lap_2p | pbg | expert | 66.4 | N16 T0.25 | 24/24 | w1 |
| anti_grav/lap_2p | pbg | noisy | 3.2 | N4 T1 | 24/24 | w1 |
| anti_grav/lap_2p | pbg | random | 0.0 | N1 T1 | 24/24 | w1 |
| anti_grav/navigation | pbf | expert | 40.0 | N1 T1 | 24/24 | w1 |
| anti_grav/navigation | pbf | noisy | 49.6 | N32 T0.25 | 24/24 | w1 |
| anti_grav/navigation | pbf | random | 20.8 | N16 T1 | 24/24 | w1 |
| anti_grav/navigation | pbg | expert | 32.8 | N16 T1 | 19/24 | w1 |
| anti_grav/navigation | pbg | noisy | 43.2 | N16 T1 | 24/24 | w1 |
| anti_grav/navigation | pbg | random | 3.2 | N32 T0.5 | 24/24 | w1 |
| ice/navigation | pbg | expert | 12.0 | N1 T0 | 1/24 | w1 |

## 전체 mean

| dataset | hiql | tr_hiql | pbg | pbf | trl |
|---|---:|---:|---:|---:|---:|
| 1k mean | 11.0 | 17.2 | 11.7 | 18.4 | 15.0 |
| 10k mean (완료분) | 20.1 | 33.8 | 31.0 | 31.5 | 11.3 |

## 1k

### 1k · expert

| env/task | hiql | tr_hiql | pbg | pbf | trl |
|---|---:|---:|---:|---:|---:|
| anti_grav/lap_2p | 16.0 | 35.2 [w1] | 66.4* [w1] | 60.0* [w1] | 0.8 |
| anti_grav/navigation | 19.2 | 67.2 [w1] | 32.8* [w1] | 40.0* [w1] | 30.4 |
| grav/lap_2p | 40.0 | 30.4 [w1] | 28.8 [w1] | 33.6 [w1] | 7.2 |
| grav/navigation | 21.6 | 92.8 [w1] | 28.0 [w1] | 26.4 [w1] | 2.4 |
| ice/lap_2p | 5.6 | 11.2 [w1] | 0.0 [w1] | 5.6 [w1] | 5.6 |
| ice/navigation | 24.0 | 28.0 [w1] | 12.0* [w1] | 32.8 [w1] | 27.2 |
| blackhole/swingby | 16.0 | 12.0 [w1] | 28.0 [w1] | 28.0 [w1] | 24.0 |
| planet/swingby | 24.0 | 20.0 [w1] | 0.0 [w1] | 28.0 [w1] | 76.0 |
| **mean** | **20.8** | **37.1** | **24.5** | **31.8** | **21.7** |

### 1k · noisy

| env/task | hiql | tr_hiql | pbg | pbf | trl |
|---|---:|---:|---:|---:|---:|
| anti_grav/lap_2p | 0.0 | 0.0 [w1] | 3.2* [w1] | 2.4* [w1] | 8.8 |
| anti_grav/navigation | 9.6 | 41.6 [w1] | 43.2* [w1] | 49.6* [w1] | 9.6 |
| grav/lap_2p | 0.0 | 0.0 [w1] | 0.8 [w1] | 0.0 [w1] | 4.0 |
| grav/navigation | 0.0 | 0.0 [w1] | 4.8 [w1] | 16.0 [w1] | 4.8 |
| ice/lap_2p | 0.0 | 0.0 [w1] | 0.0 [w1] | 0.0 [w1] | 8.8 |
| ice/navigation | 39.2 | 9.6 [w1] | 13.6 [w1] | 28.0 [w1] | 17.6 |
| blackhole/swingby | 0.0 | 0.0 [w1] | 4.0 [w1] | 4.0 [w1] | 16.0 |
| planet/swingby | 28.0 | 20.0 [w1] | 0.0 [w1] | 32.0 [w1] | 72.0 |
| **mean** | **9.6** | **8.9** | **8.7** | **16.5** | **17.7** |

### 1k · random

| env/task | hiql | tr_hiql | pbg | pbf | trl |
|---|---:|---:|---:|---:|---:|
| anti_grav/lap_2p | 0.0 | 0.0 [w1] | 0.0* [w1] | 0.0* [w1] | 0.0 |
| anti_grav/navigation | 0.0 | 0.0 [w1] | 3.2* [w1] | 20.8* [w1] | 0.0 |
| grav/lap_2p | 0.0 | 0.0 [w1] | 0.0 [w1] | 0.0 [w1] | 0.0 |
| grav/navigation | 0.0 | 13.6 [w1] | 10.4 [w1] | 30.4 [w1] | 15.2 |
| ice/lap_2p | 0.0 | 0.0 [w1] | 0.0 [w1] | 0.0 [w1] | 0.0 |
| ice/navigation | 20.0 | 28.0 [w1] | 1.6 [w1] | 3.2 [w1] | 0.8 |
| blackhole/swingby | 0.0 | 4.0 [w1] | 0.0 [w1] | 0.0 [w1] | 4.0 |
| planet/swingby | 0.0 | 0.0 [w1] | 0.0 [w1] | 0.0 [w1] | 24.0 |
| **mean** | **2.5** | **5.7** | **1.9** | **6.8** | **5.5** |

## 10k

### 10k · expert

| env/task | hiql | tr_hiql | pbg | pbf | trl |
|---|---:|---:|---:|---:|---:|
| anti_grav/lap_2p | 56.8 | 100.0 [w1] | 94.4 [w0] | 100.0 [w0] | 28.0 |
| anti_grav/navigation | 76.0 | 96.8 [w1] | 48.8 [w1] | 60.8 [w1] | 30.4 |
| grav/lap_2p | 68.8 | 55.2 [w1] | 39.2 [w1] | 48.0 [w1] | 28.0 |
| grav/navigation | 60.0 | 95.2 [w1] | 88.8 [w1] | 93.6 [w1] | 16.0 |
| ice/lap_2p | 24.8 | 52.8 [w1] | 52.8 [w1] | 56.0 [w1] | 24.8 |
| ice/navigation | 88.8 | 44.8 [w1] | 69.6 [w1] | 45.6 [w1] | 35.2 |
| **mean** | **62.5** | **74.1** | **65.6** | **67.3** | **27.1** |

### 10k · noisy

| env/task | hiql | tr_hiql | pbg | pbf | trl |
|---|---:|---:|---:|---:|---:|
| anti_grav/lap_2p | 20.0 | 16.0 [w0] | 40.8 [w0] | 45.6 [w0] | 2.4 |
| anti_grav/lap_4p | 0.8 | — | — | — | 10.4 |
| anti_grav/lap_8p | 15.2 | — | — | — | — |
| anti_grav/navigation | 46.4 | 98.4 [w1] | 78.4 [w1] | 100.0 [w1] | 20.8 |
| grav/lap_2p | 9.6 | 40.0 [w1] | 52.0 [w1] | 21.6 [w1] | 0.0 |
| grav/lap_4p | 0.0 | 24.8 [w0] | 48.0 [w0] | 22.4 [w0] | 0.8 |
| grav/lap_8p | 3.2 | 0.0 [w0] | 40.8 [w0] | 0.8 [w0] | 2.4 |
| grav/navigation | 8.0 | 58.4 [w1] | 18.4 [w1] | 68.8 [w1] | 5.6 |
| ice/lap_2p | 8.8 | 25.6 [w1] | 0.0 [w1] | 7.2 [w1] | 0.0 |
| ice/lap_4p | 0.0 | 20.0 [w0] | 0.0 [w0] | 0.0 [w0] | 0.0 |
| ice/lap_8p | 0.0 | 0.0 [w0] | 0.0 [w0] | 0.0 [w0] | 0.0 |
| ice/navigation | 61.6 | 57.6 [w1] | 34.4 [w1] | 55.2 [w1] | 46.4 |
| **mean** | **14.5** | **34.1** | **31.3** | **32.2** | **8.1** |

### 10k · random

| env/task | hiql | tr_hiql | pbg | pbf | trl |
|---|---:|---:|---:|---:|---:|
| anti_grav/lap_2p | 0.0 | 0.0 [w0] | — | — | 0.0 |
| anti_grav/lap_4p | 0.0 | 0.0 [w0] | — | — | 0.0 |
| anti_grav/navigation | 3.2 | 45.6 [w1] | 19.2 [w1→0@4k] | 29.6 [w1→0@6k] | 7.2 |
| grav/lap_2p | 0.0 | 0.0 [w1] | 0.0 [w1] | 0.0 [w1] | 0.0 |
| grav/lap_4p | 0.0 | 0.0 [w0] | 0.8 [w0] | 0.0 [w0] | 0.0 |
| grav/lap_8p | 0.0 | 0.0 [w0] | 0.0 [w0] | 0.0 [w0] | 0.0 |
| grav/navigation | 10.4 | 39.2 [w1] | 8.8 [w1] | 6.4 [w1] | 16.8 |
| ice/lap_2p | 0.0 | 0.0 [w1] | 0.0 [w1] | 0.0 [w1] | 0.0 |
| ice/lap_4p | 0.0 | 0.0 [w0] | 0.0 [w0] | 0.0 [w0] | 0.0 |
| ice/lap_8p | 0.0 | 0.0 [w0] | 0.0 [w0] | 0.0 [w0] | 0.0 |
| ice/navigation | 21.6 | 42.4 [w1] | 40.8 [w1] | 24.8 [w1] | 40.0 |
| **mean** | **3.2** | **11.6** | **7.7** | **6.8** | **5.8** |
