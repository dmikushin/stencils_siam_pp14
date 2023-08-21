# initial config
set term pdf dashed
set output 'roofline_gtx680m.pdf'
set nokey
set grid layerdefault linetype 0 linewidth 1.000, linetype 0 linewidth 1.000

set xlabel "Arithmetic intensity (flops/byte)"
set ylabel "Attaindable GFlops/s (SP)"

# sets log base 2 scale for both axis
set logscale x 2
set logscale y 2

# label offsets
L_MEM_X=0.125
L_MEM_ANG=36

# range of each axis
MIN_X=0.1
MAX_X=256
MIN_Y=8
MAX_Y=4096
set xrange [MIN_X:MAX_X]
set yrange [MIN_Y:MAX_Y]

# GTX680M constants
NUM_CORES = 1344
CORE_FREQ = 0.758
NUM_SM = 7
NUM_SFU = 32

# ceilings
C_NOFMA_NOSFU = NUM_CORES * CORE_FREQ
C_FMA_NOSFU = NUM_CORES * CORE_FREQ * 2
C_FMA_SFU = NUM_CORES * CORE_FREQ * 2 + NUM_SM * NUM_SFU * CORE_FREQ

# MEM CONSTATS
C_ALL_CORES = 1
PEAK_MEM_BW=115.2
NUM_CHANNELS=2
# first ceiling, without multiple memory channels
C_NO_MULTI_CHANNEL = NUM_CHANNELS

# FUNCTIONS
gpu_ceiling(x, y) = min(mem_roof(x), y)
mem_roof(x) = x * PEAK_MEM_BW
min(x, y) = (x < y) ? x : y
max(x, y) = (x > y) ? x : y

# line style
LINE_CEIL=2
LINE_POINTER=3
POINT_CIRCLE=4
LINE_TEST01_CUDA=11
LINE_TEST01_PGI=12
LINE_TEST02_CUDA=21
LINE_TEST02_PGI=22
LINE_TEST03_CUDA=31
LINE_TEST03_PGI=32
LINE_TEST04_CUDA=41
LINE_TEST04_PGI=42
set style line LINE_CEIL lt 1 lw 3 lc rgb "blue"
set style line LINE_POINTER lt 1 lw 3 lc rgb "black"
set style line POINT_CIRCLE lc rgb 'black' pt 7
set style line LINE_TEST01_CUDA lt 2 lw 3 lc rgb "red"
set style line LINE_TEST01_PGI lt 3 lw 3 lc rgb "red"
set style line LINE_TEST02_CUDA lt 2 lw 3 lc rgb "orange"
set style line LINE_TEST02_PGI lt 3 lw 3 lc rgb "orange"
set style line LINE_TEST03_CUDA lt 2 lw 3 lc rgb "purple"
set style line LINE_TEST03_PGI lt 3 lw 3 lc rgb "purple"
set style line LINE_TEST04_CUDA lt 2 lw 3 lc rgb "brown"
set style line LINE_TEST04_PGI lt 3 lw 3 lc rgb "brown"

# data

WAVE13PT_CUDA_INTENS=0.5405238075
WAVE13PT_CUDA_GFLOPS=43.6204269416
WAVE13PT_PGI_INTENS=1.1001557117
WAVE13PT_PGI_GFLOPS=39.8057000077

TRICUBIC_CUDA_INTENS=7.2703611099
TRICUBIC_CUDA_GFLOPS=132.7111242363
TRICUBIC_PGI_INTENS=7.4302790773
TRICUBIC_PGI_GFLOPS=124.8867757598

LBMD3Q19_CUDA_INTENS=1.1278874757
LBMD3Q19_CUDA_GFLOPS=77.3068011976

DIVERGENCE_CUDA_INTENS=0.3672428601
DIVERGENCE_CUDA_GFLOPS=35.2707061503
DIVERGENCE_PGI_INTENS=0.3013504537
DIVERGENCE_PGI_GFLOPS=21.4143573056

# PLOTS:
# Roofline with its own color, the rest 15 colors are used for different tests
# Line style is the type of target
set multiplot

# Legend (emulated)
set object rectangle at screen 0.228,0.825 size char 11.2, char 4.2 fillcolor rgb 'white' fillstyle solid border lc rgb "black" lw 1
LEGPOS = 0.90
set label "wave13pt cuda\nwave13pt pgi\ntricubic cuda\ntricubic pgi\nlbmd3q19 cuda\nlbmd3q19 pgi\ndivergence cuda\ndivergence pgi" at screen 0.21,LEGPOS font "Times-Roman, 3"
set arrow from screen 0.155,LEGPOS to screen 0.20,LEGPOS nohead ls LINE_TEST01_CUDA
set object rectangle at screen 0.177,LEGPOS size char 0.7, char 0.4 fillcolor rgb 'red' fillstyle solid border lw 0
LEGPOS = LEGPOS - 0.0215
set arrow from screen 0.155,LEGPOS to screen 0.20,LEGPOS nohead ls LINE_TEST01_PGI
set object circle at screen 0.177,LEGPOS size char 0.4, char 0.4 fillcolor rgb 'red' fillstyle solid border lw 0
LEGPOS = LEGPOS - 0.0215
set arrow from screen 0.155,LEGPOS to screen 0.20,LEGPOS nohead ls LINE_TEST02_CUDA
set object rectangle at screen 0.177,LEGPOS size char 0.7, char 0.4 fillcolor rgb 'orange' fillstyle solid border lw 0
LEGPOS = LEGPOS - 0.0215
set arrow from screen 0.155,LEGPOS to screen 0.20,LEGPOS nohead ls LINE_TEST02_PGI
set object circle at screen 0.177,LEGPOS size char 0.4, char 0.4 fillcolor rgb 'orange' fillstyle solid border lw 0
LEGPOS = LEGPOS - 0.0215
set arrow from screen 0.155,LEGPOS to screen 0.20,LEGPOS nohead ls LINE_TEST03_CUDA
set object rectangle at screen 0.177,LEGPOS size char 0.7, char 0.4 fillcolor rgb 'purple' fillstyle solid border lw 0
LEGPOS = LEGPOS - 0.0215
set arrow from screen 0.155,LEGPOS to screen 0.20,LEGPOS nohead ls LINE_TEST03_PGI
set object circle at screen 0.177,LEGPOS size char 0.4, char 0.4 fillcolor rgb 'purple' fillstyle solid border lw 0
LEGPOS = LEGPOS - 0.0215
set arrow from screen 0.155,LEGPOS to screen 0.20,LEGPOS nohead ls LINE_TEST04_CUDA
set object rectangle at screen 0.177,LEGPOS size char 0.7, char 0.4 fillcolor rgb 'brown' fillstyle solid border lw 0
LEGPOS = LEGPOS - 0.0215
set arrow from screen 0.155,LEGPOS to screen 0.20,LEGPOS nohead ls LINE_TEST04_PGI
set object circle at screen 0.177,LEGPOS size char 0.4, char 0.4 fillcolor rgb 'brown' fillstyle solid border lw 0

# Data lines
set arrow from WAVE13PT_CUDA_INTENS,MIN_Y to WAVE13PT_CUDA_INTENS,max(gpu_ceiling(WAVE13PT_CUDA_INTENS, C_FMA_NOSFU),WAVE13PT_CUDA_GFLOPS) nohead ls LINE_TEST01_CUDA
set arrow from WAVE13PT_PGI_INTENS,MIN_Y to WAVE13PT_PGI_INTENS,max(gpu_ceiling(WAVE13PT_PGI_INTENS, C_FMA_NOSFU),WAVE13PT_PGI_GFLOPS) nohead ls LINE_TEST01_PGI
set arrow from TRICUBIC_CUDA_INTENS,MIN_Y to TRICUBIC_CUDA_INTENS,max(gpu_ceiling(TRICUBIC_CUDA_INTENS, C_FMA_NOSFU),TRICUBIC_CUDA_GFLOPS) nohead ls LINE_TEST02_CUDA
set arrow from TRICUBIC_PGI_INTENS,MIN_Y to TRICUBIC_PGI_INTENS,max(gpu_ceiling(TRICUBIC_PGI_INTENS, C_FMA_NOSFU),TRICUBIC_PGI_GFLOPS) nohead ls LINE_TEST02_PGI
set arrow from LBMD3Q19_CUDA_INTENS,MIN_Y to LBMD3Q19_CUDA_INTENS,max(gpu_ceiling(LBMD3Q19_CUDA_INTENS, C_FMA_NOSFU),LBMD3Q19_CUDA_GFLOPS) nohead ls LINE_TEST03_CUDA
set arrow from DIVERGENCE_CUDA_INTENS,MIN_Y to DIVERGENCE_CUDA_INTENS,max(gpu_ceiling(DIVERGENCE_CUDA_INTENS, C_FMA_NOSFU),DIVERGENCE_CUDA_GFLOPS) nohead ls LINE_TEST04_CUDA
set arrow from DIVERGENCE_PGI_INTENS,MIN_Y to DIVERGENCE_PGI_INTENS,max(gpu_ceiling(DIVERGENCE_PGI_INTENS, C_FMA_NOSFU),DIVERGENCE_PGI_GFLOPS) nohead ls LINE_TEST04_PGI

# Data points
set object rectangle at WAVE13PT_CUDA_INTENS,WAVE13PT_CUDA_GFLOPS size char 1, char 0.6 fillcolor rgb 'red' fillstyle solid border lw 0
set object circle at WAVE13PT_PGI_INTENS,WAVE13PT_PGI_GFLOPS size char 0.6, char 0.6 fillcolor rgb 'red' fillstyle solid border lw 0
set object rectangle at TRICUBIC_CUDA_INTENS,TRICUBIC_CUDA_GFLOPS size char 1, char 0.6 fillcolor rgb 'orange' fillstyle solid border lw 0
set object circle at TRICUBIC_PGI_INTENS,TRICUBIC_PGI_GFLOPS size char 0.6, char 0.6 fillcolor rgb 'orange' fillstyle solid border lw 0
set object rectangle at TRICUBIC_CUDA_INTENS,TRICUBIC_CUDA_GFLOPS size char 1, char 0.6 fillcolor rgb 'orange' fillstyle solid border lw 0
set object rectangle at LBMD3Q19_CUDA_INTENS,LBMD3Q19_CUDA_GFLOPS size char 1, char 0.6 fillcolor rgb 'purple' fillstyle solid border lw 0
set object rectangle at DIVERGENCE_CUDA_INTENS,DIVERGENCE_CUDA_GFLOPS size char 1, char 0.6 fillcolor rgb 'brown' fillstyle solid border lw 0
set object circle at DIVERGENCE_PGI_INTENS,DIVERGENCE_PGI_GFLOPS size char 0.6, char 0.6 fillcolor rgb 'brown' fillstyle solid border lw 0

# No FMA, no SFU roofline
set label 4 "No FMA, no SFU" at (MAX_X - 64),(C_NOFMA_NOSFU * 1.2) right
plot gpu_ceiling(x, C_NOFMA_NOSFU) ls LINE_CEIL

# FMA, no SFU roofline
set label 5 "FMA, no SFU" at 11,(C_FMA_NOSFU * 0.9) right
plot gpu_ceiling(x, C_FMA_NOSFU) ls LINE_CEIL
set arrow from 11.5,(C_FMA_NOSFU * 0.9) to (C_FMA_NOSFU / PEAK_MEM_BW - 1), (C_FMA_NOSFU - 24) ls LINE_POINTER
set arrow from 13.5,(C_FMA_SFU * 1.2) to (C_FMA_SFU / PEAK_MEM_BW - 1), (C_FMA_SFU + 24) ls LINE_POINTER

# FMA, SFU roofline
set label 6 "FMA, SFU" at 13,(C_FMA_SFU * 1.3) right
plot gpu_ceiling(x, C_FMA_SFU) ls LINE_CEIL

unset multiplot

