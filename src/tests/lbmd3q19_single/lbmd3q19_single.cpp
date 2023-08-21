// TODO split single kernel into two
// TODO replace % with special case handling
// TODO naive CUDA version

#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timing.h"

#if defined(_OPENACC)
#include <openacc.h>
#include "openacc_profiling.h"
#endif

#if defined(__CUDACC__)
#include "cuda_profiling.h"
#endif

// Memory alignment, for vectorization on MIC.
// 4096 should be best for memory transfers over PCI-E.
#define MEMALIGN 4096

#define _1 ((real) 1.0)
#define _3 ((real) 3.0)
#define _3_2 ((real) 1.5)
#define _9_2 ((real) 4.5)

// lattice constants
#define _1_3 ((real) (1.0 / 3.0))
#define _1_18 ((real) (1.0 / 18.0))
#define _1_36 ((real) (1.0 / 36.0))

#define swap(a, b) do { real tmp = b; b = a; a = tmp; } while(0)

#define _A3(array, is, iy, ix) (array[(ix) + nx * ((iy) + ny * (is))])
#define _A4(array, is, iy, ix, il) (array[il][(ix) + nx * ((iy) + ny * (is))])

#include "genlbm.h"

struct step
{
	#if defined(__CUDACC__) && !defined(_PPCG)
	__device__
	#endif
	__attribute__((always_inline)) static void run(
		const int nx, const int ny, const int ns,
		const real tau,
		const real* const __restrict__ bodyforce,
		const real* const* const __restrict__ fp, /* input grid (prev density) */
		real* const* const __restrict__ fn, /* output grid (new density) */
		real* const __restrict__ ux, real* const __restrict__ uy, real* const __restrict__ uz, /* outputs (velocity field) */
		real* const __restrict__ rho, /* output (density) */
		const int* const __restrict__ solid, /* 0 if the cell at (i,j,k) isn't solid, !=0 if it is */
		int k, int j, int i, int front, int back, int bottom, int top, int left, int right)
	{
		if (_A3(solid, k, j, i) != 0) return;

		// calculate rho and ux, uy
		real _rho = (real) 0;
		for (int l = 0; l < 19; l++)
			_rho += _A4(fp, k, j, i, l);
		_A3(rho, k, j, i) = _rho;

		real _ux = (_A4(fp, k,j,i, 1) + _A4(fp, k,j,i, 2) + _A4(fp, k,j,i, 8) - _A4(fp, k,j,i, 4) - _A4(fp, k,j,i, 5) -
			    _A4(fp, k,j,i, 6) + _A4(fp, k,j,i,15) + _A4(fp, k,j,i,18) - _A4(fp, k,j,i,16) - _A4(fp, k,j,i,17)) /
			    _rho + tau * bodyforce[0];
		real _uy = (_A4(fp, k,j,i, 2) + _A4(fp, k,j,i, 3) + _A4(fp, k,j,i, 4) - _A4(fp, k,j,i, 6) - _A4(fp, k,j,i, 7) -
			    _A4(fp, k,j,i, 8) + _A4(fp, k,j,i, 9) + _A4(fp, k,j,i,14) - _A4(fp, k,j,i,11) - _A4(fp, k,j,i,12)) /
			    _rho + tau * bodyforce[1];
		real _uz = (_A4(fp, k,j,i, 9) + _A4(fp, k,j,i,10) + _A4(fp, k,j,i,11) - _A4(fp, k,j,i,12) - _A4(fp, k,j,i,13) -
			    _A4(fp, k,j,i,14) + _A4(fp, k,j,i,15) + _A4(fp, k,j,i,16) - _A4(fp, k,j,i,17) - _A4(fp, k,j,i,18)) /
			    _rho + tau * bodyforce[2];

		_A3(ux, k,j,i) = _ux;
		_A3(uy, k,j,i) = _uy;
		_A3(uz, k,j,i) = _uz;

		real _ux2 = _ux * _ux;
		real _uy2 = _uy * _uy;
		real _uz2 = _uz * _uz;
		real _uxy2 = _ux2 + _uy2;
		real _uxz2 = _ux2 + _uz2;
		real _uyz2 = _uy2 + _uz2;
		real _uxyz2 = _uxy2 + _uz2;
		real _uxy = 2 * _ux * _uy;
		real _uxz = 2 * _ux * _uz;
		real _uyz = 2 * _uy * _uz;

		// stream (stencil operation)
		real _3_2_uxyz2 = (real) 1.5 * _uxyz2;
		_A4(fn,          k,j,i, 0) = _A4(fp, k,j,i, 0) - (_A4(fp, k,j,i, 0) -
			( _1_3 * _rho * (_1 - _3_2 * _uxyz2))) / tau;
		_A4(fn,       back,j,i, 1) = _A4(fp, k,j,i, 1) - (_A4(fp, k,j,i, 1) -
			(_1_18 * _rho * (_1 + _3 * _ux          + _9_2 * (_ux2        ) - _3_2_uxyz2))) / tau;
		_A4(fn,     back,top,i, 2) = _A4(fp, k,j,i, 2) - (_A4(fp, k,j,i, 2) -
			(_1_36 * _rho * (_1 + _3 * (+_ux + _uy) + _9_2 * (_uxy2 + _uxy) - _3_2_uxyz2))) / tau;
		_A4(fn,        k,top,i, 3) = _A4(fp, k,j,i, 3) - (_A4(fp, k,j,i, 3) -
			(_1_18 * _rho * (_1 + _3 * (_uy       ) + _9_2 * (_uy2        ) - _3_2_uxyz2))) / tau;
		_A4(fn,    front,top,i, 4) = _A4(fp, k,j,i, 4) - (_A4(fp, k,j,i, 4) -
			(_1_36 * _rho * (_1 + _3 * (-_ux + _uy) + _9_2 * (_uxy2 - _uxy) - _3_2_uxyz2))) / tau;
		_A4(fn,      front,j,i, 5) = _A4(fp, k,j,i, 5) - (_A4(fp, k,j,i, 5) -
			(_1_18 * _rho * (_1 - _3 * (_ux       ) + _9_2 * (_ux2        ) - _3_2_uxyz2))) / tau;
		_A4(fn, front,bottom,i, 6) = _A4(fp, k,j,i, 6) - (_A4(fp, k,j,i, 6) -
			(_1_36 * _rho * (_1 + _3 * (-_ux - _uy) + _9_2 * (_uxy2 + _uxy) - _3_2_uxyz2))) / tau;
		_A4(fn,     k,bottom,i, 7) = _A4(fp, k,j,i, 7) - (_A4(fp, k,j,i, 7) -
			(_1_18 * _rho * (_1 - _3 * (_uy       ) + _9_2 * (_uy2        ) - _3_2_uxyz2))) / tau;
		_A4(fn,  back,bottom,i, 8) = _A4(fp, k,j,i, 8) - (_A4(fp, k,j,i, 8) -
			(_1_36 * _rho * (_1 + _3 * (+_ux - _uy) + _9_2 * (_uxy2 - _uxy) - _3_2_uxyz2))) / tau;
		_A4(fn,    k,top,right, 9) = _A4(fp, k,j,i, 9) - (_A4(fp, k,j,i, 9) -
			(_1_36 * _rho * (_1 + _3 * (+_uy + _uz) + _9_2 * (_uyz2 + _uyz) - _3_2_uxyz2))) / tau;
		_A4(fn,      k,j,right,10) = _A4(fp, k,j,i,10) - (_A4(fp, k,j,i,10) -
			(_1_18 * _rho * (_1 + _3 * (+_uz      ) + _9_2 * (_uz2        ) - _3_2_uxyz2))) / tau;
		_A4(fn, k,bottom,right,11) = _A4(fp, k,j,i,11) - (_A4(fp, k,j,i,11) -
			(_1_36 * _rho * (_1 + _3 * (-_uy + _uz) + _9_2 * (_uyz2 - _uyz) - _3_2_uxyz2))) / tau;
		_A4(fn,  k,bottom,left,12) = _A4(fp, k,j,i,12) - (_A4(fp, k,j,i,12) -
			(_1_36 * _rho * (_1 + _3 * (-_uy - _uz) + _9_2 * (_uyz2 + _uyz) - _3_2_uxyz2))) / tau;
		_A4(fn,       k,j,left,13) = _A4(fp, k,j,i,13) - (_A4(fp, k,j,i,13) -
			(_1_18 * _rho * (_1 + _3 * (-_uz      ) + _9_2 * (_uz2        ) - _3_2_uxyz2))) / tau;
		_A4(fn,     k,top,left,14) = _A4(fp, k,j,i,14) - (_A4(fp, k,j,i,14) -
			(_1_36 * _rho * (_1 + _3 * (+_uy - _uz) + _9_2 * (_uyz2 - _uyz) - _3_2_uxyz2))) / tau;
		_A4(fn,   back,j,right,15) = _A4(fp, k,j,i,15) - (_A4(fp, k,j,i,15) -
			(_1_36 * _rho * (_1 + _3 * (+_ux + _uz) + _9_2 * (_uxz2 + _uxz) - _3_2_uxyz2))) / tau;
		_A4(fn,  front,j,right,16) = _A4(fp, k,j,i,16) - (_A4(fp, k,j,i,16) -
			(_1_36 * _rho * (_1 + _3 * (-_ux + _uz) + _9_2 * (_uxz2 - _uxz) - _3_2_uxyz2))) / tau;
		_A4(fn,   front,j,left,17) = _A4(fp, k,j,i,17) - (_A4(fp, k,j,i,17) -
			(_1_36 * _rho * (_1 + _3 * (-_ux - _uz) + _9_2 * (_uxz2 + _uxz) - _3_2_uxyz2))) / tau;
		_A4(fn,    back,j,left,18) = _A4(fp, k,j,i,18) - (_A4(fp, k,j,i,18) -
			(_1_36 * _rho * (_1 + _3 * (+_ux - _uz) + _9_2 * (_uxz2 - _uxz) - _3_2_uxyz2))) / tau;
	}
};

struct swap
{
	#if defined(__CUDACC__) && !defined(_PPCG)
	__device__
	#endif
	__attribute__((always_inline)) static void run(
		const int nx, const int ny, const int ns,
		const real tau,
		const real* const __restrict__ bodyforce,
		const real* const* const __restrict__ fp, /* input grid (prev density) */
		real* const* const __restrict__ fn, /* output grid (new density) */
		real* const __restrict__ ux, real* const __restrict__ uy, real* const __restrict__ uz, /* outputs (velocity field) */
		real* const __restrict__ rho, /* output (density) */
		const int* const __restrict__ solid, /* 0 if the cell at (i,j,k) isn't solid, !=0 if it is */
		int k, int j, int i, int front, int back, int bottom, int top, int left, int right)
	{
		if (_A3(solid, k,j,i) == 0) return;

		// boundary
		swap(_A4(fn, k,j,i, 1), _A4(fn, k,j,i, 5));
		swap(_A4(fn, k,j,i, 2), _A4(fn, k,j,i, 6));
		swap(_A4(fn, k,j,i, 3), _A4(fn, k,j,i, 7));
		swap(_A4(fn, k,j,i, 4), _A4(fn, k,j,i, 8));
		swap(_A4(fn, k,j,i, 9), _A4(fn, k,j,i,12));
		swap(_A4(fn, k,j,i,10), _A4(fn, k,j,i,13));
		swap(_A4(fn, k,j,i,14), _A4(fn, k,j,i,11));
		swap(_A4(fn, k,j,i,15), _A4(fn, k,j,i,17));
		swap(_A4(fn, k,j,i,18), _A4(fn, k,j,i,16));

		/*// stream boundary
		_A4(fn,       back,j,i, 1) = _A4(fn, k,j,i, 1);
		_A4(fn,     back,top,i, 2) = _A4(fn, k,j,i, 2);
		_A4(fn,        k,top,i, 3) = _A4(fn, k,j,i, 3);
		_A4(fn,    front,top,i, 4) = _A4(fn, k,j,i, 4);
		_A4(fn,      front,j,i, 5) = _A4(fn, k,j,i, 5);
		_A4(fn, front,bottom,i, 6) = _A4(fn, k,j,i, 6);
		_A4(fn,     k,bottom,i, 7) = _A4(fn, k,j,i, 7);
		_A4(fn,  back,bottom,i, 8) = _A4(fn, k,j,i, 8);
		_A4(fn,    k,top,right, 9) = _A4(fn, k,j,i, 9);
		_A4(fn,      k,j,right,10) = _A4(fn, k,j,i,10);
		_A4(fn, k,bottom,right,11) = _A4(fn, k,j,i,11);
		_A4(fn,  k,bottom,left,12) = _A4(fn, k,j,i,12);
		_A4(fn,       k,j,left,13) = _A4(fn, k,j,i,13);
		_A4(fn,     k,top,left,14) = _A4(fn, k,j,i,14);
		_A4(fn,   back,j,right,15) = _A4(fn, k,j,i,15);
		_A4(fn,  front,j,right,16) = _A4(fn, k,j,i,16);
		_A4(fn,   front,j,left,17) = _A4(fn, k,j,i,17);
		_A4(fn,    back,j,left,18) = _A4(fn, k,j,i,18);*/
	}
};

template<typename function>
#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
__global__
#endif
// Periodic boundaries are handled separately for speed.
void lbmd3q19_domain(
	const int nx, const int ny, const int ns,
#if defined(__CUDACC__) && !defined(_PPCG)
	kernelgen_cuda_config_t config,
#endif
	const real tau,
	const real* const __restrict__ bodyforce,
	const real* const* const __restrict__ fp, /* input grid (prev density) */
	real* const* const __restrict__ fn, /* output grid (new density) */
	real* const __restrict__ ux, real* const __restrict__ uy, real* const __restrict__ uz, /* outputs (velocity field) */
	real* const __restrict__ rho, /* output (density) */
	const int* const __restrict__ solid /* 0 if the cell at (i,j,k) isn't solid, !=0 if it is */)
{
#if defined(__CUDACC__) && !defined(_PPCG)
	#define i_stride (config.strideDim.x)
	#define j_stride (config.strideDim.y)
	#define k_stride (config.strideDim.z)
	#define k_offset (blockIdx.z * blockDim.z + threadIdx.z)
	#define j_offset (blockIdx.y * blockDim.y + threadIdx.y)
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
	#define k_increment k_stride
	#define j_increment j_stride
	#define i_increment i_stride
#else
	#define k_offset 0
	#define j_offset 0
	#define i_offset 0
	#define k_increment 1
	#define j_increment 1
	#define i_increment 1
#endif
#if defined(_PATUS)
	real* dummy;
	#pragma omp parallel
	wave13pt_patus(&dummy, w0, w1, w2, m0, m1, m2, nx, ny, ns);
#else
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(ns), present( \
		bodyforce[0:3], fp[0:19][0:szarray], fn[0:19][0:szarray], \
		ux[0:szarray], uy[0:szarray], uz[0:szarray], rho[0:szarray], solid[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
#if defined(_PPCG)
	#pragma scop
#endif
	for (int k = 1 + k_offset; k < ns - 1; k += k_increment)
	{
#if defined(_OPENACC)
		#pragma acc loop independent
		for (int j = 1 + j_offset; j < ny - 1; j += j_increment)
		{
#else
		for (int j = j_offset; j < ny; j += j_increment)
		{
			if ((j < 1) || (j >= ny - 2)) continue;
#endif
#if defined(_OPENACC)
			#pragma acc loop independent vector(512)
#endif
			for (int i = i_offset; i < nx; i += i_increment)
			{

#if defined(__CUDACC__) && !defined(_PPCG) && ((defined(__CUDA_SHMEM2D__) && defined(__CUDA_SHMEM2DREG1D__)) || \
	(defined(__CUDA_SHMEM1D__) && defined(__CUDA_SHMEM1DREG1D__)))

				// Following Paulius Micikevicius: 2D slice in shared memory,
				// values of third dimension column are shared through small
				// array, that compiler maps on registers.
				int k = 1 + k_offset;
#if defined(__CUDA_VECTORIZE2__)
				real2 w1_reg[5];
#else
				real w1_reg[5];
#endif
				if (k < ns - 2)
				{
					_R1D(w1, 1) = _A(w1, k-2, j, i);
					_R1D(w1, 2) = _A(w1, k-1, j, i);
					_R1D(w1, 3) = _A(w1, k, j, i);
					_R1D(w1, 4) = _A(w1, k+1, j, i);
				}
				for ( ; k < ns - 2; k += k_increment)
				{
					_R1D(w1, 0) = _R1D(w1, 1);
					_R1D(w1, 1) = _R1D(w1, 2);
					_R1D(w1, 2) = _R1D(w1, 3);
					_R1D(w1, 3) = _R1D(w1, 4);
					_R1D(w1, 4) = _A(w1, k+2, j, i);

#endif

#if defined(__CUDACC__) && !defined(_PPCG) && defined(__CUDA_SHMEM1D__)

#if defined(__CUDA_VECTORIZE2__)
					int i_shm = 2 * threadIdx.x;
#else
					int i_shm = threadIdx.x;
#endif

#if 0 // BEGIN TODO
					_S1D(w1, i_shm) = _A(w1, k, j, i);
					if (i_shm < 2)
					{
						_S1D(w1, i_shm - 2) = _A(w1, k, j, i - 2);
#if defined(__CUDA_VECTORIZE2__)
						if (i + 2 * blockDim.x < nx)
							_S1D(w1, i_shm + 2 * blockDim.x) = _A(w1, k, j, i + 2 * blockDim.x);
#else
						if (i + blockDim.x < nx)
							_S1D(w1, i_shm + blockDim.x) = _A(w1, k, j, i + blockDim.x);
#endif
					}
					__syncthreads();
#endif // END TODO
					if ((i < 1) || (i >= nx - 1)) continue;

#if !defined(__CUDA_SHMEM1DREG1D__)

#if defined(__CUDA_VECTORIZE2__)

#if 0 // BEGIN TODO
					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _S1D(w1, i_shm);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_j_ip2 = _S1D(w1, i_shm + 2);
					real2 w1_k_j_im2 = _S1D(w1, i_shm - 2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _A(w1, k, j+1, i);
					real2 w1_k_jm1_i = _A(w1, k, j-1, i);
					real2 w1_k_jp2_i = _A(w1, k, j+2, i);
					real2 w1_k_jm2_i = _A(w1, k, j-2, i);
					
					real2 w1_kp1_j_i = _A(w1, k+1, j, i);
					real2 w1_km1_j_i = _A(w1, k-1, j, i);
					real2 w1_kp2_j_i = _A(w1, k+2, j, i);
					real2 w1_km2_j_i = _A(w1, k-2, j, i);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					_A(w2, k, j, i) =  m0 * _S1D(w1, i_shm) - _A(w0, k, j, i) +

						m1 * (
							_S1D(w1, i_shm + 1) + _S1D(w1, i_shm - 1) +
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						m2 * (
							_S1D(w1, i_shm + 2) + _S1D(w1, i_shm - 2) +
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));
#endif // END TODO

#endif

#else

#if defined(__CUDA_VECTORIZE2__)

#if 0 // BEGIN TODO
					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _S1D(w1, i_shm);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_j_ip2 = _S1D(w1, i_shm + 2);
					real2 w1_k_j_im2 = _S1D(w1, i_shm - 2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _A(w1, k, j+1, i);
					real2 w1_k_jm1_i = _A(w1, k, j-1, i);
					real2 w1_k_jp2_i = _A(w1, k, j+2, i);
					real2 w1_k_jm2_i = _A(w1, k, j-2, i);
					
					real2 w1_kp1_j_i = _R1D(w1, 3);
					real2 w1_km1_j_i = _R1D(w1, 1);
					real2 w1_kp2_j_i = _R1D(w1, 4);
					real2 w1_km2_j_i = _R1D(w1, 0);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					_A(w2, k, j, i) =  m0 * _S1D(w1, i_shm) - _A(w0, k, j, i) +

						m1 * (
							_S1D(w1, i_shm + 1) + _S1D(w1, i_shm - 1) +
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_R1D(w1, 3) + _R1D(w1, 1)) +
						m2 * (
							_S1D(w1, i_shm + 2) + _S1D(w1, i_shm - 2) +
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_R1D(w1, 4) + _R1D(w1, 0));
#endif // END TODO

#endif

#endif
					
#elif defined(__CUDACC__) && !defined(_PPCG) && defined(__CUDA_SHMEM2D__)

#if defined(__CUDA_VECTORIZE2__)
					int i_shm = 2 * threadIdx.x, j_shm = threadIdx.y;
#else
					int i_shm = threadIdx.x, j_shm = threadIdx.y;
#endif

#if 0 // BEGIN TODO					
					_S2D(w1, j_shm, i_shm) = _A(w1, k, j, i);
					if (j_shm < 2)
					{
						_S2D(w1, j_shm - 2, i_shm) = _A(w1, k, j - 2, i);
						if (j + blockDim.y < ny)
							_S2D(w1, j_shm + blockDim.y, i_shm) = _A(w1, k, j + blockDim.y, i);
					}
					if (i_shm < 2)
					{
						_S2D(w1, j_shm, i_shm - 2) = _A(w1, k, j, i - 2);
#if defined(__CUDA_VECTORIZE2__)
						if (i + 2 * blockDim.x < nx)
							_S2D(w1, j_shm, i_shm + 2 * blockDim.x) = _A(w1, k, j, i + 2 * blockDim.x);
#else
						if (i + blockDim.x < nx)
							_S2D(w1, j_shm, i_shm + blockDim.x) = _A(w1, k, j, i + blockDim.x);
#endif
					}
					__syncthreads();
#endif // END TODO
					if ((j < 1) || (j >= ny - 1)) continue;
					if ((i < 1) || (i >= nx - 1)) continue;

#if !defined(__CUDA_SHMEM2DREG1D__)

#if defined(__CUDA_VECTORIZE2__)

#if 0 // BEGIN TODO
					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _S2D(w1, j_shm, i_shm);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_j_ip2 = _S2D(w1, j_shm, i_shm + 2);
					real2 w1_k_j_im2 = _S2D(w1, j_shm, i_shm - 2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _S2D(w1, j_shm + 1, i_shm);
					real2 w1_k_jm1_i = _S2D(w1, j_shm - 1, i_shm);
					real2 w1_k_jp2_i = _S2D(w1, j_shm + 2, i_shm);
					real2 w1_k_jm2_i = _S2D(w1, j_shm - 2, i_shm);
					
					real2 w1_kp1_j_i = _A(w1, k+1, j, i);
					real2 w1_km1_j_i = _A(w1, k-1, j, i);
					real2 w1_kp2_j_i = _A(w1, k+2, j, i);
					real2 w1_km2_j_i = _A(w1, k-2, j, i);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					_A(w2, k, j, i) =  m0 * _S2D(w1, j_shm, i_shm) - _A(w0, k, j, i) +

						m1 * (
							_S2D(w1, j_shm, i_shm + 1) + _S2D(w1, j_shm, i_shm - 1) +
							_S2D(w1, j_shm + 1, i_shm) + _S2D(w1, j_shm - 1, i_shm) +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						m2 * (
							_S2D(w1, j_shm, i_shm + 2) + _S2D(w1, j_shm, i_shm - 2) +
							_S2D(w1, j_shm + 2, i_shm) + _S2D(w1, j_shm - 2, i_shm) +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));
#endif // END TODO

#endif

#else

#if defined(__CUDA_VECTORIZE2__)

#if 0 // BEGIN TODO
					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _S2D(w1, j_shm, i_shm);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_j_ip2 = _S2D(w1, j_shm, i_shm + 2);
					real2 w1_k_j_im2 = _S2D(w1, j_shm, i_shm - 2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _S2D(w1, j_shm + 1, i_shm);
					real2 w1_k_jm1_i = _S2D(w1, j_shm - 1, i_shm);
					real2 w1_k_jp2_i = _S2D(w1, j_shm + 2, i_shm);
					real2 w1_k_jm2_i = _S2D(w1, j_shm - 2, i_shm);
					
					real2 w1_kp1_j_i = _R1D(w1, 3);
					real2 w1_km1_j_i = _R1D(w1, 1);
					real2 w1_kp2_j_i = _R1D(w1, 4);
					real2 w1_km2_j_i = _R1D(w1, 0);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					_A(w2, k, j, i) =  m0 * _S2D(w1, j_shm, i_shm) - _A(w0, k, j, i) +

						m1 * (
							_S2D(w1, j_shm, i_shm + 1) + _S2D(w1, j_shm, i_shm - 1) +
							_S2D(w1, j_shm + 1, i_shm) + _S2D(w1, j_shm - 1, i_shm) +
							_R1D(w1, 3) + _R1D(w1, 1)) +
						m2 * (
							_S2D(w1, j_shm, i_shm + 2) + _S2D(w1, j_shm, i_shm - 2) +
							_S2D(w1, j_shm + 2, i_shm) + _S2D(w1, j_shm - 2, i_shm) +
							_R1D(w1, 4) + _R1D(w1, 0));
#endif // END TODO

#endif

#endif

#else

#if !defined(__CUDA_SHUFFLE__)

					if ((i < 1) || (i >= nx - 1)) continue;

#endif

#if defined(__CUDA_VECTORIZE2__)

#if defined(__CUDA_SHUFFLE__)

#if 0 // BEGIN TODO
					real2 val = _A(w1, k, j, i+2);
					real2 result; result.x = m2 * val.x; result.y = m2 * val.y;
					real swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if (laneid == 0)
						val.x = _AS(w1, k, j, i+1);
					result.x += m1 * val.x; result.y += m1 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1))
						val.x = _AS(w1, k, j, i);
					result.x += m0 * val.x; result.y += m0 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2))
						val.x =_AS(w1, k, j, i-1);
					result.x += m1 * val.x; result.y += m1 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2) || (laneid == 3))
						val.x = _AS(w1, k, j, i-2);
					result.x += m2 * val.x; result.y += m2 * val.y;

					if ((i < 1) || (i >= nx - 1)) continue;

					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_jp1_i = _A(w1, k, j+1, i);
					real2 w1_k_jm1_i = _A(w1, k, j-1, i);
					real2 w1_k_jp2_i = _A(w1, k, j+2, i);
					real2 w1_k_jm2_i = _A(w1, k, j-2, i);
					
					real2 w1_kp1_j_i = _A(w1, k+1, j, i);
					real2 w1_km1_j_i = _A(w1, k-1, j, i);
					real2 w1_kp2_j_i = _A(w1, k+2, j, i);
					real2 w1_km2_j_i = _A(w1, k-2, j, i);

					(*w2_k_j_i).x =  result.x - w0_k_j_i.x +

						m1 * (
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  result.y - w0_k_j_i.y +

						m1 * (
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _A(w1, k, j, i);
					real2 w0_k_j_i = _A(w0, k, j, i);
					
					real2 w1_k_j_ip2 = _A(w1, k, j, i+2);
					real2 w1_k_j_im2 = _A(w1, k, j, i-2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _A(w1, k, j+1, i);
					real2 w1_k_jm1_i = _A(w1, k, j-1, i);
					real2 w1_k_jp2_i = _A(w1, k, j+2, i);
					real2 w1_k_jm2_i = _A(w1, k, j-2, i);
					
					real2 w1_kp1_j_i = _A(w1, k+1, j, i);
					real2 w1_km1_j_i = _A(w1, k-1, j, i);
					real2 w1_kp2_j_i = _A(w1, k+2, j, i);
					real2 w1_km2_j_i = _A(w1, k-2, j, i);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#endif

#elif defined(__CUDA_SHUFFLE__)

#if 0 // BEGIN TODO
					real val = _A(w1, k, j, i+2);
					real result = m2 * val;
					val = __shfl_up(val, 1);
					if (laneid == 0)
						val = _A(w1, k, j, i+1);
					result += m1 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1))
						val = _A(w1, k, j, i);
					result += m0 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2))
						val = _A(w1, k, j, i-1);
					result += m1 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2) || (laneid == 3))
						val = _A(w1, k, j, i-2);
					result += m2 * val;

					if ((i < 1) || (i >= nx - 1)) continue;

					_A(w2, k, j, i) = result - _A(w0, k, j, i) +

						m1 * (
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						m2 * (
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));
#endif // END TODO

#else

					function::run(nx, ny, ns, tau, bodyforce, fp, fn, ux, uy, uz, rho, solid,
						k, j, i, k - 1, k + 1, j - 1, j + 1, i - 1, i + 1);

#endif

#endif

#if defined(__CUDA_SHMEM2DREG1D__) || defined(__CUDA_SHMEM1DREG1D__)

				} // k-loop

#endif

			} // i-loop
		} // j-loop
		
#if !defined(__CUDA_SHMEM2DREG1D__) && !defined(__CUDA_SHMEM1DREG1D__)

	} // k-loop

#endif

#if defined(_PPCG)
	#pragma endscop
#endif

#endif
}

template<typename function>
#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
__global__
#endif
// D3Q19 Lattice Boltzmann: calculate XY periodic boundaries on CPU
void lbmd3q19_boundaries_xy(
	const int nx, const int ny, const int ns,
#if defined(__CUDACC__) && !defined(_PPCG)
	kernelgen_cuda_config_t config,
#endif
	const real tau,
	const real* const __restrict__ bodyforce,
	const real* const* const __restrict__ fp, /* input grid (prev density) */
	real* const* const __restrict__ fn, /* output grid (new density) */
	real* const __restrict__ ux, real* const __restrict__ uy, real* const __restrict__ uz, /* outputs (velocity field) */
	real* const __restrict__ rho, /* output (density) */
	const int* const __restrict__ solid /* 0 if the cell at (i,j,k) isn't solid, !=0 if it is */)
{
#if defined(__CUDACC__) && !defined(_PPCG)
	#define i_stride (config.strideDim.x)
	#define j_stride (config.strideDim.y)
	#define k_stride (config.strideDim.z)
	#define k_offset (blockIdx.z * blockDim.z + threadIdx.z)
	#define j_offset (blockIdx.y * blockDim.y + threadIdx.y)
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
	#define k_increment k_stride
	#define j_increment j_stride
	#define i_increment i_stride
#else
	#define k_offset 0
	#define j_offset 0
	#define i_offset 0
	#define k_increment 1
	#define j_increment 1
	#define i_increment 1
#endif
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(ns), present( \
		bodyforce[0:3], fp[0:19][0:szarray], fn[0:19][0:szarray], \
		ux[0:szarray], uy[0:szarray], uz[0:szarray], rho[0:szarray], solid[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
#if defined(_PPCG)
	#pragma scop
#endif
	for (int j = j_offset; j < ny; j += j_increment)
	{
#if !defined(_OPENACC)
		const int bottom = (j + ny - 1) % ny;
		const int top = (j + 1) % ny;
#endif
		for (int i = i_offset; i < nx; i += i_increment)
		{
#if defined(_OPENACC)
			const int bottom = (j + ny - 1) % ny;
			const int top = (j + 1) % ny;
#endif
			{
				const int k = 0;
				const int front = (k + ns - 1) % ns;
				const int back = (k + 1) % ns;

				const int left = (i + nx - 1) % nx;
				const int right = (i + 1) % nx;
			
				function::run(nx, ny, ns, tau, bodyforce, fp, fn, ux, uy, uz, rho, solid,
					k, j, i, front, back, bottom, top, left, right);
			}
			{
				const int k = ns - 1;
				const int front = (k + ns - 1) % ns;
				const int back = (k + 1) % ns;

				const int left = (i + nx - 1) % nx;
				const int right = (i + 1) % nx;
			
				function::run(nx, ny, ns, tau, bodyforce, fp, fn, ux, uy, uz, rho, solid,
					k, j, i, front, back, bottom, top, left, right);
			}
		}
	}
#if defined(_PPCG)
	#pragma endscop
#endif
}

template<typename function>
#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
__global__
#endif
// D3Q19 Lattice Boltzmann: calculate XZ periodic boundaries on CPU
void lbmd3q19_boundaries_xz(
	const int nx, const int ny, const int ns,
#if defined(__CUDACC__) && !defined(_PPCG)
	kernelgen_cuda_config_t config,
#endif
	const real tau,
	const real* const __restrict__ bodyforce,
	const real* const* const __restrict__ fp, /* input grid (prev density) */
	real* const* const __restrict__ fn, /* output grid (new density) */
	real* const __restrict__ ux, real* const __restrict__ uy, real* const __restrict__ uz, /* outputs (velocity field) */
	real* const __restrict__ rho, /* output (density) */
	const int* const __restrict__ solid /* 0 if the cell at (i,j,k) isn't solid, !=0 if it is */)
{
#if defined(__CUDACC__) && !defined(_PPCG)
	#define i_stride (config.strideDim.x)
	#define j_stride (config.strideDim.y)
	#define k_stride (config.strideDim.z)
	#define k_offset (blockIdx.z * blockDim.z + threadIdx.z)
	#define j_offset (blockIdx.y * blockDim.y + threadIdx.y)
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
	#define k_increment k_stride
	#define j_increment j_stride
	#define i_increment i_stride
#else
	#define k_offset 0
	#define j_offset 0
	#define i_offset 0
	#define k_increment 1
	#define j_increment 1
	#define i_increment 1
#endif
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(ns), present( \
		bodyforce[0:3], fp[0:19][0:szarray], fn[0:19][0:szarray], \
		ux[0:szarray], uy[0:szarray], uz[0:szarray], rho[0:szarray], solid[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
#if defined(_PPCG)
	#pragma scop
#endif
	for (int k = k_offset; k < ns; k += k_increment)
	{
#if !defined(_OPENACC)
		const int front = (k + ns - 1) % ns;
		const int back = (k + 1) % ns;
#endif
		for (int i = i_offset; i < nx; i += i_increment)
		{
#if defined(_OPENACC)
			const int front = (k + ns - 1) % ns;
			const int back = (k + 1) % ns;
#endif
			{
				const int j = 0;
				const int bottom = (j + ny - 1) % ny;
				const int top = (j + 1) % ny;

				const int left = (i + nx - 1) % nx;
				const int right = (i + 1) % nx;
				
				function::run(nx, ny, ns, tau, bodyforce, fp, fn, ux, uy, uz, rho, solid,
					k, j, i, front, back, bottom, top, left, right);
			}
			{
				const int j = ny - 1;
				const int bottom = (j + ny - 1) % ny;
				const int top = (j + 1) % ny;

				const int left = (i + nx - 1) % nx;
				const int right = (i + 1) % nx;
				
				function::run(nx, ny, ns, tau, bodyforce, fp, fn, ux, uy, uz, rho, solid,
					k, j, i, front, back, bottom, top, left, right);
			}
		}
	}
}

template<typename function>
#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
__global__
#endif
// D3Q19 Lattice Boltzmann: calculate YZ periodic boundaries on CPU
void lbmd3q19_boundaries_yz(
	const int nx, const int ny, const int ns,
#if defined(__CUDACC__) && !defined(_PPCG)
	kernelgen_cuda_config_t config,
#endif
	const real tau,
	const real* const __restrict__ bodyforce,
	const real* const* const __restrict__ fp, /* input grid (prev density) */
	real* const* const __restrict__ fn, /* output grid (new density) */
	real* const __restrict__ ux, real* const __restrict__ uy, real* const __restrict__ uz, /* outputs (velocity field) */
	real* const __restrict__ rho, /* output (density) */
	const int* const __restrict__ solid /* 0 if the cell at (i,j,k) isn't solid, !=0 if it is */)
{
#if defined(__CUDACC__) && !defined(_PPCG)
	#define i_stride (config.strideDim.x)
	#define j_stride (config.strideDim.y)
	#define k_stride (config.strideDim.z)
	#define k_offset (blockIdx.z * blockDim.z + threadIdx.z)
	#define j_offset (blockIdx.y * blockDim.y + threadIdx.y)
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
	#define k_increment k_stride
	#define j_increment j_stride
	#define i_increment i_stride
#else
	#define k_offset 0
	#define j_offset 0
	#define i_offset 0
	#define k_increment 1
	#define j_increment 1
	#define i_increment 1
#endif
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(ns), present( \
		bodyforce[0:3], fp[0:19][0:szarray], fn[0:19][0:szarray], \
		ux[0:szarray], uy[0:szarray], uz[0:szarray], rho[0:szarray], solid[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
#if defined(_PPCG)
	#pragma scop
#endif
	for (int k = k_offset; k < ns; k += k_increment)
	{
#if !defined(_OPENACC)
		const int front = (k + ns - 1) % ns;
		const int back = (k + 1) % ns;
#endif
		for (int j = 1 + j_offset; j < ny - 1; j += j_increment)
		{
#if defined(_OPENACC)
			const int front = (k + ns - 1) % ns;
			const int back = (k + 1) % ns;
#endif
			const int bottom = (j + ny - 1) % ny;
			const int top = (j + 1) % ny;

			{
				const int i = 0;
				const int left = (i + nx - 1) % nx;
				const int right = (i + 1) % nx;
				
				function::run(nx, ny, ns, tau, bodyforce, fp, fn, ux, uy, uz, rho, solid,
					k, j, i, front, back, bottom, top, left, right);
			}
			{
				const int i = nx - 1;
				const int left = (i + nx - 1) % nx;
				const int right = (i + 1) % nx;
				
				function::run(nx, ny, ns, tau, bodyforce, fp, fn, ux, uy, uz, rho, solid,
					k, j, i, front, back, bottom, top, left, right);
			}
		}
	}
}

#define parse_arg(name, arg) \
	int name = atoi(arg); \
	if (name < 0) \
	{ \
		printf("Value for " #name " is invalid: %d\n", name); \
		exit(1); \
	}

#define real_rand() (((real)(rand() / (double)RAND_MAX) - 0.5) * 2)

int main(int argc, char* argv[])
{
	if (argc != 5)
	{
		printf("Usage: %s <nx> <ny> <ns> <nt>\n", argv[0]);
		exit(1);
	}

	const char* no_timing = getenv("NO_TIMING");

#if defined(_OPENACC) || defined(__CUDACC__)
	char* regcount_fname = getenv("PROFILING_FNAME");
	if (regcount_fname)
	{
		char* regcount_lineno = getenv("PROFILING_LINENO");
		int lineno = -1;
		if (regcount_lineno)
			lineno = atoi(regcount_lineno);
		kernelgen_enable_regcount(regcount_fname, lineno);
	}
#endif

	parse_arg(nx, argv[1]);
	parse_arg(ny, argv[2]);
	parse_arg(ns, argv[3]);
	parse_arg(nt, argv[4]);

	size_t szarray = (size_t)nx * ny * ns;
	size_t szarrayb = szarray * sizeof(real);

	real bodyforce[3];
	real** fp = (real**)memalign(MEMALIGN, 19 * sizeof(real*));
	real** fn = (real**)memalign(MEMALIGN, 19 * sizeof(real*));
	for (int i = 0; i < 19; i++)
	{
		fp[i] = (real*)memalign(MEMALIGN, szarrayb);
		fn[i] = (real*)memalign(MEMALIGN, szarrayb);
	}
	real* ux = (real*)memalign(MEMALIGN, szarrayb);
	real* uy = (real*)memalign(MEMALIGN, szarrayb);
	real* uz = (real*)memalign(MEMALIGN, szarrayb);
	real* rho = (real*)memalign(MEMALIGN, szarrayb);
	int* solid = (int*)memalign(MEMALIGN, szarray * sizeof(int));

	for (int i = 0; i < 19; i++)
	{	
		memset(fp[i], 0, szarrayb);
		memset(fn[i], 0, szarrayb);
	}
	memset(ux, 0, szarrayb);
	memset(uy, 0, szarrayb);
	memset(uz, 0, szarrayb);
	memset(rho, 0, szarrayb);
	memset(solid, 0, szarrayb);

	if (!fp || !fn || !ux || !uy || !uz || !rho || !solid)
	{
		printf("Error allocating memory for arrays: %p, %p, %p, %p, %p, %p, %p\n",
			fp, fn, ux, uy, uz, rho, solid);
		exit(1);
	}

	// generate input data
	real tau = real_rand();
	genlbm(nx, ny, ns, bodyforce,
		fp, fn, ux, uy, uz, rho, solid);

	real mean1 = 0.0f;
	for (int i = 0; i < szarray; i++)
		for (int k = 0; k < 19; k++)
			mean1 += fp[k][i] + fn[k][i];
	real mean2 = 0.0f;
	for (int i = 0; i < szarray; i++)
		mean2 += ux[i] + uy[i] + uz[i] + rho[i];
	real mean = tau + bodyforce[0] + bodyforce[1] + bodyforce[2] +
		mean1 / (szarray * 19) / 2 + mean2 / szarray / 6;
	printf("initial mean = %f\n", mean);

	//
	// MIC or OPENACC or CUDA:
	//
	// 1) Perform an empty offload, that should strip
	// the initialization time from further offloads.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec init_s, init_f;
#if defined(_MIC)
	get_time(&init_s);
	#pragma offload target(mic) \
		nocopy(bodyforce:length(3) alloc_if(0) free_if(0)), \
		nocopy(fp:length(szarrayf) alloc_if(0) free_if(0)), \
		nocopy(fn:length(szarrayf) alloc_if(0) free_if(0)), \
		nocopy(ux:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(uy:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(uz:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(rho:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(solid:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&init_f);
#endif
#if defined(_OPENACC)
	get_time(&init_s);
#if defined(__PGI)
	acc_init(acc_device_nvidia);
#else
	acc_init(acc_device_gpu);
#endif
	get_time(&init_f);
#endif
#if defined(__CUDACC__)
	get_time(&init_s);
	int count = 0;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
	get_time(&init_f);
#endif
	double init_t = get_time_diff((struct timespec*)&init_s, (struct timespec*)&init_f);
	if (!no_timing) printf("init time = %f sec\n", init_t);
#endif

	//
	// MIC or OPENACC or CUDA:
	//
	// 2) Allocate data on device, but do not copy anything.
	//
#if defined(_MIC) || defined(_OPENACC) || (defined(__CUDACC__) && !defined(_PPCG))
	volatile struct timespec alloc_s, alloc_f;
#if defined(_MIC)
	get_time(&alloc_s);
	#pragma offload target(mic) \
		nocopy(bodyforce:length(3) alloc_if(1) free_if(0)), \
		nocopy(fp:length(szarrayf) alloc_if(1) free_if(0)), \
		nocopy(fn:length(szarrayf) alloc_if(1) free_if(0)), \
		nocopy(ux:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(uy:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(uz:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(rho:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(solid:length(szarray) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create ( \
		bodyforce[0:3], fp[0:19][0:szarray], fn[0:19][0:szarray], \
		ux[0:szarray], uy[0:szarray], uz[0:szarray], rho[0:szarray], solid[0:szarray])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&alloc_s);
	real *bodyforce_dev = NULL, **fp_dev = NULL, **fn_dev = NULL;
	real *ux_dev = NULL, *uy_dev = NULL, *uz_dev = NULL, *rho_dev = NULL;
	int *solid_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&bodyforce_dev, 3 * sizeof(real)));
	CUDA_SAFE_CALL(cudaMalloc(&fp_dev, 19 * (sizeof(real*) + szarrayb)));
	CUDA_SAFE_CALL(cudaMalloc(&fn_dev, 19 * (sizeof(real*) + szarrayb)));
	CUDA_SAFE_CALL(cudaMalloc(&ux_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&uy_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&uz_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&rho_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&solid_dev, szarray * sizeof(int)));
	get_time(&alloc_f);
#endif
	double alloc_t = get_time_diff((struct timespec*)&alloc_s, (struct timespec*)&alloc_f);
	if (!no_timing) printf("device buffer alloc time = %f sec\n", alloc_t);
#endif

	//
	// MIC or OPENACC or CUDA:
	//
	// 3) Transfer data from host to device and leave it there,
	// i.e. do not allocate deivce memory buffers.
	//
#if defined(_MIC) || defined(_OPENACC) || (defined(__CUDACC__) && !defined(_PPCG))
	volatile struct timespec load_s, load_f;
#if defined(_MIC)
	get_time(&load_s);
	#pragma offload target(mic) \
		in(bodyforce:length(3) alloc_if(0) free_if(0)), \
		in(fp:length(szarrayf) alloc_if(0) free_if(0)), \
		in(fn:length(szarrayf) alloc_if(0) free_if(0)), \
		in(ux:length(szarray) alloc_if(0) free_if(0)), \
		in(uy:length(szarray) alloc_if(0) free_if(0)), \
		in(uz:length(szarray) alloc_if(0) free_if(0)), \
		in(rho:length(szarray) alloc_if(0) free_if(0)), \
		in(solid:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device( \
		bodyforce[0:3], fp[0:19][0:szarray], fn[0:19][0:szarray], \
		ux[0:szarray], uy[0:szarray], uz[0:szarray], rho[0:szarray], solid[0:szarray])
	get_time(&load_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(bodyforce_dev, bodyforce, 3 * sizeof(real), cudaMemcpyHostToDevice));
	for (int i = 0; i < 19; i++)
	{
		real* ptr1 = (real*)(fp_dev + 19) + szarray * i;
		CUDA_SAFE_CALL(cudaMemcpy(&fp_dev[i], &ptr1, sizeof(real*), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(ptr1, fp[i], szarrayb, cudaMemcpyHostToDevice));
		real* ptr2 = (real*)(fn_dev + 19) + szarray * i;
		CUDA_SAFE_CALL(cudaMemcpy(&fn_dev[i], &ptr2, sizeof(real*), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(ptr2, fn[i], szarrayb, cudaMemcpyHostToDevice));
	}
	CUDA_SAFE_CALL(cudaMemcpy(ux_dev, ux, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(uy_dev, uy, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(uz_dev, uz, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(rho_dev, rho, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(solid_dev, solid, szarray * sizeof(int), cudaMemcpyHostToDevice));

	get_time(&load_f);
#endif
	double load_t = get_time_diff((struct timespec*)&load_s, (struct timespec*)&load_f);
	if (!no_timing) printf("data load time = %f sec (%f GB/sec)\n",
		load_t, (2 * szarrayb * 19 + 5 * szarrayb + 3 * sizeof(real)) / (load_t * 1024 * 1024 * 1024));
#endif

	//
	// 4) Perform data processing iterations, keeping all data
	// on device.
	//
	int idxs[] = { 0, 1 };
	volatile struct timespec compute_s, compute_f;
#if defined(__CUDACC__) && !defined(_PPCG)
	dim3 gridDim, blockDim, strideDim;
	kernelgen_cuda_config_t config;
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
#if (defined(__CUDA_SHMEM1D__) || defined(__CUDA_SHMEM2D__)) && \
	(defined(__CUDA_VECTORIZE2__) || defined(__CUDA_VECTORIZE4__))
	CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#endif
	if (sizeof(real) == sizeof(double))
		CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#if defined(__CUDA_SHMEM1D__)
	kernelgen_cuda_configure_gird(1, nx, ny, ns, &config);
	kernelgen_cuda_configure_shmem(&config, 1, sizeof(real),
		config.blockDim.x + STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT,
		STENCIL_BOUNDARY_LEFT);
#elif defined(__CUDA_SHMEM2D__)
	kernelgen_cuda_configure_gird(2, nx, ny, ns, &config);
	kernelgen_cuda_configure_shmem(&config, 1, sizeof(real),
		(config.blockDim.x + STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT + SHMEM_BANK_SHIFT) *
		(config.blockDim.y + STENCIL_BOUNDARY_TOP + STENCIL_BOUNDARY_BOTTOM), STENCIL_BOUNDARY_TOP *
		(config.blockDim.x + STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT + SHMEM_BANK_SHIFT) +
		STENCIL_BOUNDARY_LEFT);
#else
	kernelgen_cuda_configure_gird(1, nx, ny, ns, &config);
#endif
#endif
	get_time(&compute_s);
#if defined(_MIC)
	#pragma offload target(mic) \
		nocopy(bodyforce:length(3) alloc_if(0) free_if(0)), \
		nocopy(fp:length(szarrayf) alloc_if(0) free_if(0)), \
		nocopy(fn:length(szarrayf) alloc_if(0) free_if(0)), \
		nocopy(ux:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(uy:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(uz:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(rho:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(solid:length(szarray) alloc_if(0) free_if(0))
#endif
	{
#if !defined(__CUDACC__) || defined(_PPCG)
		real **fpp = fp, **fnp = fn;
#else
		real **fpp = fp_dev, **fnp = fn_dev;
#endif
		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__) || defined(_PPCG)
			lbmd3q19_domain<step>(
				nx, ny, ns, tau, bodyforce,
				fpp, fnp, ux, uy, uz, rho, solid);
			lbmd3q19_boundaries_xy<step>(
				nx, ny, ns, tau, bodyforce,
				fpp, fnp, ux, uy, uz, rho, solid);
			lbmd3q19_boundaries_xz<step>(
				nx, ny, ns, tau, bodyforce,
				fpp, fnp, ux, uy, uz, rho, solid);
			lbmd3q19_boundaries_yz<step>(
				nx, ny, ns, tau, bodyforce,
				fpp, fnp, ux, uy, uz, rho, solid);

			lbmd3q19_domain<swap>(
				nx, ny, ns, tau, bodyforce,
				fpp, fnp, ux, uy, uz, rho, solid);
			lbmd3q19_boundaries_xy<swap>(
				nx, ny, ns, tau, bodyforce,
				fpp, fnp, ux, uy, uz, rho, solid);
			lbmd3q19_boundaries_xz<swap>(
				nx, ny, ns, tau, bodyforce,
				fpp, fnp, ux, uy, uz, rho, solid);
			lbmd3q19_boundaries_yz<swap>(
				nx, ny, ns, tau, bodyforce,
				fpp, fnp, ux, uy, uz, rho, solid);
#else
			lbmd3q19_domain<step>
				<<<config.gridDim, config.blockDim, config.szshmem>>>(
				nx, ny, ns,
				config,
				tau, bodyforce_dev,
				fpp, fnp, ux_dev, uy_dev, uz_dev, rho_dev, solid_dev);
			CUDA_SAFE_CALL(cudaGetLastError());
			{
				kernelgen_cuda_config_t config_xy = config;
				config_xy.gridDim.z = 1; config_xy.blockDim.z = 1;
				lbmd3q19_boundaries_xy<step>
					<<<config_xy.gridDim, config_xy.blockDim, config_xy.szshmem>>>(
					nx, ny, ns,
					config_xy,
					tau, bodyforce_dev,
					fpp, fnp, ux_dev, uy_dev, uz_dev, rho_dev, solid_dev);
			}
			CUDA_SAFE_CALL(cudaGetLastError());
			{
				kernelgen_cuda_config_t config_xz = config;
				config_xz.gridDim.y = 1; config_xz.blockDim.y = 1;
				lbmd3q19_boundaries_xz<step>
					<<<config_xz.gridDim, config_xz.blockDim, config_xz.szshmem>>>(
					nx, ny, ns,
					config_xz,
					tau, bodyforce_dev,
					fpp, fnp, ux_dev, uy_dev, uz_dev, rho_dev, solid_dev);
			}
			CUDA_SAFE_CALL(cudaGetLastError());
			{
				kernelgen_cuda_config_t config_yz = config;
				config_yz.gridDim.x = 1; config_yz.blockDim.x = 1;
				lbmd3q19_boundaries_yz<step>
					<<<config_yz.gridDim, config_yz.blockDim, config_yz.szshmem>>>(
					nx, ny, ns,
					config_yz,
					tau, bodyforce_dev,
					fpp, fnp, ux_dev, uy_dev, uz_dev, rho_dev, solid_dev);
			}
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			lbmd3q19_domain<swap>
				<<<config.gridDim, config.blockDim, config.szshmem>>>(
				nx, ny, ns,
				config,
				tau, bodyforce_dev,
				fpp, fnp, ux_dev, uy_dev, uz_dev, rho_dev, solid_dev);
			CUDA_SAFE_CALL(cudaGetLastError());
			{
				kernelgen_cuda_config_t config_xy = config;
				config_xy.gridDim.z = 1; config_xy.blockDim.z = 1;
				lbmd3q19_boundaries_xy<swap>
					<<<config_xy.gridDim, config_xy.blockDim, config_xy.szshmem>>>(
					nx, ny, ns,
					config_xy,
					tau, bodyforce_dev,
					fpp, fnp, ux_dev, uy_dev, uz_dev, rho_dev, solid_dev);
			}
			CUDA_SAFE_CALL(cudaGetLastError());
			{
				kernelgen_cuda_config_t config_xz = config;
				config_xz.gridDim.y = 1; config_xz.blockDim.y = 1;
				lbmd3q19_boundaries_xz<swap>
					<<<config_xz.gridDim, config_xz.blockDim, config_xz.szshmem>>>(
					nx, ny, ns,
					config_xz,
					tau, bodyforce_dev,
					fpp, fnp, ux_dev, uy_dev, uz_dev, rho_dev, solid_dev);
			}
			CUDA_SAFE_CALL(cudaGetLastError());
			{
				kernelgen_cuda_config_t config_yz = config;
				config_yz.gridDim.x = 1; config_yz.blockDim.x = 1;
				lbmd3q19_boundaries_yz<swap>
					<<<config_yz.gridDim, config_yz.blockDim, config_yz.szshmem>>>(
					nx, ny, ns,
					config_yz,
					tau, bodyforce_dev,
					fpp, fnp, ux_dev, uy_dev, uz_dev, rho_dev, solid_dev);
			}
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
			real** f = fpp; fpp = fnp; fnp = f;
			int idx = idxs[0]; idxs[0] = idxs[1]; idxs[1] = idx;
		}
	}
	get_time(&compute_f);
	double compute_t = get_time_diff((struct timespec*)&compute_s, (struct timespec*)&compute_f);
	if (!no_timing) printf("compute time = %f sec\n", compute_t);

#if !defined(__CUDACC__) || defined(_PPCG)
	real** f[] = { fp, fn }; 
	fp = f[idxs[0]]; fn = f[idxs[1]];
#else
	real** f[] = { fp_dev, fn_dev }; 
	fp_dev = f[idxs[0]]; fn_dev = f[idxs[1]];
#if defined(__CUDA_SHMEM1D__) || defined(__CUDA_SHMEM2D__)
	kernelgen_cuda_config_dispose(&config);
#endif
#endif

	//
	// MIC or OPENACC or CUDA:
	//
	// 5) Transfer output data right from device to host.
	//
#if defined(_MIC) || defined(_OPENACC) || (defined(__CUDACC__) && !defined(_PPCG))
	volatile struct timespec save_s, save_f;
#if defined(_MIC)
	get_time(&save_s);
	#pragma offload target(mic) \
		out(bodyforce:length(3) alloc_if(0) free_if(0)), \
		out(fp:length(szarrayf) alloc_if(0) free_if(0)), \
		out(fn:length(szarrayf) alloc_if(0) free_if(0)), \
		out(ux:length(szarray) alloc_if(0) free_if(0)), \
		out(uy:length(szarray) alloc_if(0) free_if(0)), \
		out(uz:length(szarray) alloc_if(0) free_if(0)), \
		out(rho:length(szarray) alloc_if(0) free_if(0)), \
		out(solid:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&save_f);
#endif
#if defined(_OPENACC)
	get_time(&save_s);
	#pragma acc update host ( \
		bodyforce[0:3], fp[0:19][0:szarray], fn[0:19][0:szarray], \
		ux[0:szarray], uy[0:szarray], uz[0:szarray], rho[0:szarray], solid[0:szarray])
	get_time(&save_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&save_s);
	for (int i = 0; i < 19; i++)
	{
		real* ptr1 = (real*)(fp_dev + 19) + szarray * i;
		CUDA_SAFE_CALL(cudaMemcpy(fp[i], ptr1, szarrayb, cudaMemcpyDeviceToHost));
		real* ptr2 = (real*)(fn_dev + 19) + szarray * i;
		CUDA_SAFE_CALL(cudaMemcpy(fn[i], ptr2, szarrayb, cudaMemcpyDeviceToHost));
	}
	CUDA_SAFE_CALL(cudaMemcpy(ux, ux_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(uy, uy_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(uz, uz_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(rho, rho_dev, szarrayb, cudaMemcpyDeviceToHost));
	get_time(&save_f);
#endif
	double save_t = get_time_diff((struct timespec*)&save_s, (struct timespec*)&save_f);
	if (!no_timing) printf("data save time = %f sec (%f GB/sec)\n", save_t,
		(szarrayb * 19 * 2 + 4 * szarrayb) / (save_t * 1024 * 1024 * 1024));
#endif

	//
	// MIC or OPENACC or CUDA:
	//
	// 6) Deallocate device data buffers.
	// OPENACC does not seem to have explicit deallocation.
	//
#if defined(_OPENACC)
	}
#endif
#if defined(_MIC) || (defined(__CUDACC__) && !defined(_PPCG))
	volatile struct timespec free_s, free_f;
#if defined(_MIC)
	get_time(&free_s);
	#pragma offload target(mic) \
		nocopy(bodyforce:length(3) alloc_if(0) free_if(1)), \
		nocopy(fp:length(szarrayf) alloc_if(0) free_if(1)), \
		nocopy(fn:length(szarrayf) alloc_if(0) free_if(1)), \
		nocopy(ux:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(uy:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(uz:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(rho:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(solid:length(szarray) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(bodyforce_dev));
	CUDA_SAFE_CALL(cudaFree(fp_dev));
	CUDA_SAFE_CALL(cudaFree(fn_dev));
	CUDA_SAFE_CALL(cudaFree(ux_dev));
	CUDA_SAFE_CALL(cudaFree(uy_dev));
	CUDA_SAFE_CALL(cudaFree(uz_dev));
	CUDA_SAFE_CALL(cudaFree(rho_dev));
	CUDA_SAFE_CALL(cudaFree(solid_dev));
	get_time(&free_f);
#endif
	double free_t = get_time_diff((struct timespec*)&free_s, (struct timespec*)&free_f);
	if (!no_timing) printf("device buffer free time = %f sec\n", free_t);
#endif

	mean1 = 0.0f;
	for (int i = 0; i < szarray; i++)
		for (int k = 0; k < 19; k++)
			mean1 += fp[k][i] + fn[k][i];
	mean2 = 0.0f;
	for (int i = 0; i < szarray; i++)
		mean2 += ux[i] + uy[i] + uz[i] + rho[i];
	mean = bodyforce[0] + bodyforce[1] + bodyforce[2] +
		mean1 / (szarray * 19) / 2 + mean2 / szarray / 6;
	printf("final mean = %f\n", mean);

	for (int i = 0; i < 19; i++)
	{
		free(fp[i]);
		free(fn[i]);
	}
	free(fp);
	free(fn);
	free(ux);
	free(uy);
	free(uz);
	free(rho);
	free(solid);

	fflush(stdout);

	return 0;
}

