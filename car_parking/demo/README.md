# CarParking fixed-task renders

| Task | Maneuver | Image |
|---|---|---|
| 1 | Lower parallel parking | `task1_parallel.png` |
| 2 | Upper parallel parking | `task2_parallel.png` |
| 3 | Reverse T-bay parking | `task3_t_reverse.png` |
| 4 | Forward T-bay parking | `task4_t_forward.png` |
| 5 | Angled parking | `task5_angled.png` |

`overview.png` contains all five tasks. Regenerate the images from the project
root with:

```bash
python -m car_parking.render_demos
```
