CudaWorker - CUDA worker is used to calculate fitness (+ dipoles) after successful mutation/crossover; it includes two kernel functions (dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b, dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b - and several their variations) that calculate the arrangements of samples, errors and dipoles; kernels are called from the CalcIndivDetailedErrAndClassDistAndDipol_V2b method (or one of its variations) where also the DT is copied to GPU and, finally, the results are received by the CPU; the in-memory DT representation is chosen in Worker.h file using ADDAPTIVE_TREE_REP and FULL_BINARY_TREE_REP defines;


Worker - a base class for external computing resources (CUDA, SPARK, etc.) to outsource the time-demanding jobs


IEvolutionaryAlg - evolutionary loop


main.cpp - start file
