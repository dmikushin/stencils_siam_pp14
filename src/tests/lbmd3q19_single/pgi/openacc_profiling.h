//===----------------------------------------------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_PROFILING_H
#define OPENACC_PROFILING_H

#ifdef __cplusplus
extern "C" {
#endif

int kernelgen_enable_regcount(char* funcname, long lineno);

int kernelgen_disable_openacc_regcount();

#ifdef __cplusplus
}
#endif

#endif // OPENACC_PROFILING_H

