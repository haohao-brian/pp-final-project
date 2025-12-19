# 15618-final-project
Proposal link: https://drive.google.com/file/d/1H6Dh6pzju2yUPYrLQwE_7uuU_qGkSM_r/view?usp=sharing

Recording link: https://drive.google.com/file/d/1zFBDPYUr_2Q_xQpMZmnw3myyCP7pd_5I/view?usp=sharing

Final Report: parallel_Project_Final_Report.pdf in this folder

### Sections
[Summary](#summary) <br/>
[Background](#background) <br/>
[Challenge](#challenge) <br/>
[Resources](#resources) <br/>
[Goals and deliverables](#goals-and-deliverables) <br/>
[Platform choice](#platform-choice) <br/>
[Schedule](#schedule) <br/>
[Milestone Report](#milestone-report) <br/>
[References](#references)

### SUMMARY
We are going to implement paralleled Fast Fourier transform algorithm on both GPU and multi-core
CPU platforms, and perform a detailed analysis of both systems’ performance characteristics.
### BACKGROUND
A fast Fourier transform (FFT) is an algorithm that computes the discrete Fourier transform (DFT)
of a sequence, or its inverse (IDFT). Fourier analysis converts a signal from its original domain
(often time or space) to a representation in the frequency domain and vice versa. [1]

Fast Fourier transforms are widely used for applications in engineering, music, science, and mathe-
matics. [1] One example is big integer multiplication: with FFT, the time complexity of big integer
multiplication can be reduced from O(n2) to O(nlogn).

Moreover, FFT can be parallelized, which can help accelerate big integer multiplication even more.
### CHALLENGE
1. FFT involves multiple iterations, which may cause synchronization over head when imple-
menting parallel version.
2. The computation requires information from different locations, which brings trouble when
trying to utilizing locality, especially when running on GPU.
3. Divergent executions happens all the time.
### RESOURCES
We are not using starter codes for our implementation of Fast Fourier transform algorithm on CPU,
but might use relevant codes for our implementation on GPU (several papers regarding this topic),
and might use relevant codes for Fast Fourier transform algorithm application if we achieves our
125 percent target.

We are using MAC computers and GHC/PSC computers for CPU/GPU implementations.
### GOALS AND DELIVERABLES
75 percent: Implement parallel FFT on CPU to achieve reasonable speed-ups, explore possibility of efficient caching and even workload distribution, analyze the performance. 

100 percent: Implement parallel FFT on CPU to achieve reasonable speed-ups, explore possibility of efficient caching and even workload distribution, analyze the performance. Implement parallel FFT on GPU and analyze the performance, determine the bottleneck.

125 percent: Achieve reasonable speed-ups for GPU implementation of FFT, and explore its application (on image/video processing).

Delivery: We will show speed-up graphs from our analysis.
### PLATFORM CHOICE
We are implementing our code in C++. MAC computer are for small to medium level CPU testing,
GHC machine are good enough for small to medium level GPU testing. If we have the time to
attempt our 125 percent goal, we might try other machines with more GPU powers.
### SCHEDULE
3.24-3.30: Understand the logic of FFT.

3.31-4.06: Write sequential code for FFT; Implement parallel CPU FFT.

4.07-4.13: Analyze parallel CPU FFT performance; Prepare for checkpoint report.

4.14-4.20: Understand research papers about parallel GPU FFT(Both); Further improve parallel CPU FFT performance(Yitao); 

4.21-4.27: Implement parallel GPU FFT(Both); Analyze parallel GPU FFT performance(Both). 

4.28-5.04: Understand research papers about parallel FFT’s application in image/voice processing(Both); Try to implement FFT’s application in image/voice processing(Both).

### Milestone Report
In the last two weeks, our group studied and coded Fast Fourier Transformation (FFT). There are different FFTs for different applications, and we have written sequential code for one of the most general form -- the one for polynomial multiplication. We have also explored CPU parallization for FFT program.

We were able to reach most of our goals. An exception is to study GPU parallization of FFT. We are confident that we will be able to reach the 75 percent goal, and we are still optimistic regarding the 100 percent goal. 

At the poster session, we will briefly explain what FFT is and its usage, then we will reveal our CPU parallaized speedup analysis, and hopefully also GPU parallaized speedup analysis. There will likely be graphs and no demo.

Our CPU parallelization is showing reasonable speedup. For example, 8 cores have around 3 times speedup. It is still not ideal and we intend to further explore CPU parallelization, but this is not trivial.

At this point we are concerned with two major issues. 

(1) We are rather unfamiliar with most forms of FFT. Understanding the logic behind them and debugging is time consuming. We already have a working version for one of the FFT, but we want to explore other FFTs as well. 

(2) We haven't fully understood the GPU parallelization of FFT. Also, the GPU parallelization will very likely require modification of our current data structures, and we are unsure of the speedup we could achieve.

### Report
multi-thread O3, if no O3, almost *8. Multithread is slightly better than openmp in speedup (slower in 1 thread, faster in 8 threads).
SIMD: first part without SIMD has high portion, preventing the total speedup to reach optimal(around *3 speedup). Second part load also cause overhead. 
One interesting problem: why is 0.06 + 8 * 0.03 smaller than single thread version. Cache miss difference observed, not sure why.
part three's for loop might need parallel version <- No improvements, leave as it is.
Possibly influence from load/store. 
(1) O3
(2) load-store
(3) cache miss


### References
[1] Wikipedia. Fast Fourier transform — Wikipedia, the free encyclopedia. http://en.wikipedia.
org/w/index.php?title=Fast\%20Fourier\%20transform&oldid=1073961290, 2022. [On-
line; accessed 22-March-2022].
[2] youtube channels: Reducible; 3Blue1Brown
[3] 4 lines source code from https://blog.csdn.net/enjoy_pascal/article/details/81478582/, used in fft.cpp, function fft_plan_dft_1d.
