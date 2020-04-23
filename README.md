## DISTRIBUTED INTERACTIVE VISUALIZATION USING GPU-OPTIMIZED SPARK

![Teaser](https://github.com/hvcl/spark_in_situ/blob/master/data/teaser.png)

### Abstract

With the advent of advances in imaging and computing technology, large-scale data acquisition and processing has become commonplace in many science and engineering disciplines. Conventional workflows for large-scale data processing usually rely on in-house or commercial software that is designed for domain-specific computing tasks. Recent advances in MapReduce, which was originally developed for batch processing textual data via a simplified programming model of the map and reduce functions, have expanded its applications to more general tasks in big-data processing, such as scientific computing and biomedical image processing. However, as shown in previous work, volume rendering and visualization using MapReduce is still considered challenging and impractical due to the disk-based, batch-processing nature of its computing model. In this paper, contrary to this common belief, we show that the MapReduce computing model can be effectively used for interactive visualization. Our proposed system is a novel extension of Spark, one of the most popular open-source MapReduce frameworks, that offers GPU-accelerated MapReduce computing. To minimize CPU-GPU communication and overcome slow, disk-based shuffle performance, the proposed system supports GPU in-memory caching and MPI-based direct communication between compute nodes. To allow for GPU-accelerated in-situ visualization using raster graphics in Spark, we leveraged the CUDA-OpenGL interoperability, resulting in faster processing speeds by several orders of magnitude compared to conventional MapReduce systems. We demonstrate the performance of our system via several volume processing and visualization tasks, such as direct volume rendering, iso-surface extraction, and numerical simulations with in-situ visualization.

### Tested system environment 
* Ubuntu 16.04
* Python 2 – essential
* Hadoop 2.7 – essential
* Spark 2.1.x – essential
* CUDA 9.0 – essential
* MVAPICH 2 – essential
* Python hdfs 2.0.x – for volume image upload to HDFS
* EGL support NVIDIA driver 


