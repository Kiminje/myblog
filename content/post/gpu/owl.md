---
title: "GPU architecture 공부: CTA scheduling 01"
date: 2022-02-15T19:45:16+09:00
categories:
- GPU
- Scheduling
tags:
- gpu
- cta
- locality
keywords:
- computer architecture
#thumbnailImage: //example.com/image.jpg
draft: true
---
# OWL: Cooperative Thread Array Aware Scheduling Techniques for Improving GPGPU Performance (ASPLOS'13, Adwait Jog)
<!--more-->

## 서론
 Co-operative Thread Array(CTA) 간 data locality가 존재하되, locaity가 무작위로 발생하게 되는 경우(e.g. graph application),
 data locality가 존재하는 CTA끼리 묶어서 실행시킴으로서 성능향상을 시킬 수 있다. 하지만 CUDA 나 OpenCL 과 같은 programming 방식으로는 한계가 명확하다. 
 data locality를 잘 활용하지 못하면 큰 병렬성을 제공하는 GPU라고 할 지라도 큰 memory latency 에 의해 성능저하가 발생하게 된다. 
 따라서 이번 논문 OWL의 4가지 방법을 통해 어떻게 data locality를 개선하고 latency hiding을 하는지 익힌다. 

 ## 도입
 GPU를 공부하는 사람들은(특히 나 같은 석사 나부랭이) textbook으로 공부하는데 한계가 분명하여 논문을 통해 교수님들, 박사님들의 Introduction에 적혀져있는 Background를 통해
 공부해나간다. 그러므로 도입부분에서는 GPU architecure의 특성을 작성한다.

 GPGPU application은 커널 단위로 나눌 수 있으며, 각 커널은 다수의 스레드들로 이루어져 있다. 스레드들은 thread block 또는 cooperative thread array(CTA)로 그룹지어진다.
 GPGPU application이 실행되면, CTA 스케줄러가 CTA 스케줄링을 이용가능한 GPGPU core(streaming multiprocessor)에서 시작하도록 한다. CTA가 코어에서 실행될 떄, 보통 'warp'라고 불리는 32개의 스레드들로 이루어진 그룹단위 로 실행된다. 
 Warp 안의 모든 스레드들은 같은 instruction을 실행하게 되며, 이러한 실행방식을 Single Instruction Multiple Thread(SIMT) 이라고 한다. 
 *논문에서 말하는 'core'는 Streaming Multiprocessor(SM)를 의미하며 GPGPU는 여러개의 SM을 가지며 한 SM은 여러개의 CUDA core를 갖고 있다.*
 
 보통 한 SM에서는 1024개 이상의 스레드들을 동시에 실행할 수 있도록 설계되어 GPGPU는 이론상 매우높은 thread-level-parallelism(TLP)를 달성할 수 있다. 하지만 코어들의 높은 inactive time으로 인해 hardware under-utilization이 발생하는 것이 문제점이며 이 논문에서는 주로 세가지를 주요 원인으로 인해 발생한다고 한다. 
 
 1) on-chip memory and register files are limiting factors on parallelism
 2) high control flow divergence
 3) inefficient scheduling mechanisms

 (1)은 GPGPU가 computing unit(CUDA core)의 비중이 memory unit(RF, cache)에 비해 크기 때문에 computing unit이 달성할 수 있는 성능에 비해 메모리 시스템이 뒷받침해주지 못한다는 의미이다. 
 (2)은 GPGPU는 CPU와 다르게 in-order processor로, warp 내에서 branch가 발생하게 되었을 때 if와 else를 차례대로 따로 처리하게 된다. 다시 말하자면, 만약 if문을 실행하는 스레드만이 계속 실행된다면 병렬로 실행하는 스레드의 수가 적어진다는 의미이다. 
 (3)은 scheduling에 주로 사용되는 round-robin(RR) 알고리즘의 경우, DRAM 의 제한된 대역폭으로 인해 발생되는 long memory latency 를 숨기기에 충분하지 않다. SIMT 아키텍처에서 Round-robin의 문제점은, 대부분의 warp가 코드를 같은 수준으로 진행하기 때문에, long memory latency가 발생하는 명령어에 도달했을 때에는 scheduling할 eligible warp가 없어 GPGPU가 stall 된다는 의미이다.
 *warp가 load instruction을 실행한 후, data를 받을 때까지 stall 되지만 '실행중인' warp 이므로 active warp이고, scheduling 되어야하는 stall 상태이기 때문에 non-eligible warp이다.* 
 *(3)의 이유로 요즘에는 RR말고 Greater-Than-Older(GTO) 알고리즘을 사용하여 스케줄링을 한다(GPGPU-Sim에서도 이 알고리즘을 기본 알고리즘으로 사용). 물론 특정 어플리케이션마다 최적의 알고리즘이 존재하겠다*
 따라서 off-chip memory bandwidth가 부족하기 때문이며, unified memory를 이용하는 CPU-GPU system에서 심해지고, core 수와 실행되는 스레드 수가 증가할수록 악화된다는 것을 유추할 수 있다.
 
 위의 문제를 해결하기 위해 우선순위를 정해 특정 CTA 그룹에 대해 CTA-aware scheduling을 하는 것을 통해 long memory fetch latency를 발생시키는 여러 요인들을 완화시키기로 한다.
 방법에는 4가지가 존재한다. 

 ## CTA-aware two-level warp scheduler
 한 코어에서 실행가능한 CTA가 N개이고, 하나의 CTA가 k개의 warp로 이루어져있다고 가정하자. 적용된 방식은 이 CTA들을 n개의 CTA로 구성된 작은 그룹으로 묶어 RR 방식으로 스케줄링한다.
 따라서 스케줄러가 하나의 그룹, *n x k*개의 warp를 선택하면, *((N-n) x k)*개의 나머지 warp는 다른 그룹으로 설정된다. 선택된 초기의 그룹의 warp들은 같은 우선순위를 받고, RR방식으로 실행된다. 실해 도중 모든 warp들이 unavailable해졌을 때, 다른 그룹으로 switching이 발생하게 되고 선택된 다른 그룹안에서 또다시 RR방식으로 스케줄링이 되도록한다. 이러한 방식을 통해 memory latency hiding이 가능하다.
 *소그룹 단위로 실행이 이루어지기 떄문에 코드의 진행속도에 차이가 있기 때문에 위의 방식이 가능하다.*

 *그럼 그룹의 크기는 어떻게 구하나요?*
 n개의 CTA는 SM의 pipeline을 충분히 사용할 수 있도록 warp수가 많아야한다. 그리고 k값은 kernel에 따라 달라지기 때문에 바꿀 수는 없다. 따라서 논문에서는 n값을 늘려가면서 체크하였다. 또한 *n x k*값은 pipeline stage 수의 배수가 되어야 한다고 한다. 

두가지의 locality
warp data locality: warp안의 스레드들은 memory coalesce를 통해 locality를 활용한다. 
intra-CTA(inter-warp) data locality: CTA안의 warp들이 특정 data block 또는 row에 접근한다면, on-chip cache에서의 data reuse를 통해 memory latency를 줄일 수 있다.

## Locality aware warp scheduling
위의 설명한 CTA-aware scheduling 방식은 L1 cache를 제대로 활용하고 있지 못한다. 다수의 CTA를 한 SM에서 처리할 때, L1 cache는 고작 16-64KB으로, cache contention으로 인해 warp들의 data reuse를 방해한다.
cache contention 문제는 capacity를 늘리는 것으로 어느정도 해결이 가능하지만 capacity를 늘리게 되면 cache access latency(address bit 증가, way수 증가)가 증가하게 되며,
cache capacity가 늘어가는 만큼, computing resource가 감소하여 architecture의 parallelism을 방해할 수 있다. 

또한, 위의 방식대로라면, 그룹 CTA의 scheduling을 RR방식으로 하기 때문에 data가 도착하여 다시 available해진 그룹 CTA를 스케줄링하지 못하기 때문에 해당 data를 cache에서 reuse하기 전에 eviction되게 되므로,
최적의 방식은 아니다. 

계속 업데이트 중..