#!/bin/sh
# nohup python3 excecute.py --radius 25 --max_gap 10  --frame_interval 15 --thr_frame 100  > ./logs/summary_and_plots_radius25_maxgap10_frameinterval5.out ;
#nohup python3 excecute.py --radius 15 --max_gap 10  --frame_interval 5 --thr_frame 100 > ./logs/summary_and_plots_radius15_maxgap10_frameinterval5.out &
# nohup python3 excecute.py --radius 15 --max_gap 10  --frame_interval 15 --thr_frame 100 > ./logs/summary_and_plots_radius15_maxgap10_frameinterval15.out &
nohup python3 excecute.py --radius 25 --max_gap 10  --frame_interval 5 --thr_frame 100 > ./logs/summary_and_plots_radius25_maxgap10_frameinterval5.out &
