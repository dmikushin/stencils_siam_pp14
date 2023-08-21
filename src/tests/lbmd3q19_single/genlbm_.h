#ifndef GENLBM_H
#define GENLBM_H

// sample simulation constants

// low  density fluid (1 for singlephase)
#define RHO_LOW         0.0734
// high density fluid (1 for singlephase)
#define RHO_HIGH        2.6429
// density of solid walls
#define RHO_BOUNDARY    0.2

#define NUM_DROPLETS    2
#define DROPLET         { 1, 1 }    // 1 for drop, -1 for bubble
#define DROPLET_POS_X   { 20, 50 }
#define DROPLET_POS_Y   { 30, 30 }
#define DROPLET_POS_Z   { 30, 30 }
#define DROPLET_RADIUS  { 15, 10 }
#define INTERFACE_WIDTH 3.0

// boundaries;
// 0=periodic; 1=HBB, set half way bounce back on respective wall
#define BOUNDARY_BOTTOM 1
#define BOUNDARY_TOP    1
#define BOUNDARY_LEFT   0
#define BOUNDARY_RIGHT  0
#define BOUNDARY_FRONT  0
#define BOUNDARY_BACK   0

// gravity force
#define BODYFORCE       2.e-5
// gravity direction (0=down 90=right 180=top 270=left)
#define BODYFORCE_DIR   0

// interparticular interaction potential (G=0 for singlephase)
#define G               -6.0

// lattice constants
#define _1_3 ((real) (1.0 / 3.0))
#define _1_18 ((real) (1.0 / 18.0))
#define _1_36 ((real) (1.0 / 36.0))


// LBM Initialization function.
void genlbm(
	const int nx, const int ny, const int ns,
	real* const __restrict__ bodyforce,  /* body force vector */
	real* const* const __restrict__ fp, /* input grid (prev density) */
	real* const* const __restrict__ fn, /* output grid (new density) */
	real* const __restrict__ ux, real* const __restrict__ uy, real* const __restrict__ uz, /* outputs (velocity field) */
	real* const __restrict__ rho, /* output (density) */
	int* const __restrict__ solid) /* 0 if the cell at (i,j,k) isn't solid, !=0 if it is */
{
	const real wi[19] = {
		_1_3 , _1_18, _1_36, _1_18, _1_36, _1_18,
		_1_36, _1_18, _1_36, _1_36, _1_18, _1_36,
		_1_36, _1_18, _1_36, _1_36, _1_36, _1_36, _1_36 };

	real droplet[] = DROPLET;
	int droplet_pos_x[] = DROPLET_POS_X;
	int droplet_pos_y[] = DROPLET_POS_Y;
	int droplet_pos_z[] = DROPLET_POS_Z;
	int droplet_radius[] = DROPLET_RADIUS;

	// initialize body_force vector
	bodyforce[0] = (real) (BODYFORCE * sin(BODYFORCE_DIR / (180.0 / M_PI)));
	bodyforce[1] = (real) (-BODYFORCE * cos(BODYFORCE_DIR / (180.0 / M_PI)));
	bodyforce[2] = (real) 0.0;

	// initialize grids
	//#pragma omp parallel for
	for (int k = 0; k < ns; k++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				// density and lattice cell types
				if ((BOUNDARY_LEFT > 0 && i == 0) || (BOUNDARY_RIGHT > 0 && i == nx - 1) ||
					(BOUNDARY_BOTTOM > 0 && j == 0) || (BOUNDARY_TOP > 0 && j == ny - 1) ||
					(BOUNDARY_FRONT > 0 && k == 0) || (BOUNDARY_BACK > 0 && k == ns - 1))
				{
					_A3(solid, k,j,i) = 1;
					_A3(rho, k,j,i) = (real) (RHO_BOUNDARY * (RHO_HIGH - RHO_LOW) + RHO_LOW);
				}
				else
				{
					_A3(solid, k,j,i) = 0;
					_A3(rho, k,j,i) = (real) RHO_LOW;

					// add droplets
					for (int l = 0; l < NUM_DROPLETS; l++)
					{
						real dx2 = i - droplet_pos_x[l];
						dx2 *= dx2;
						real dy2 = j - droplet_pos_y[l];
						dy2 *= dy2;
						real dz2 = k - droplet_pos_z[l];
						dz2 *= dz2;
						
						real radius = sqrt(dx2 + dy2 + dz2);
						real tmp = (real) (0.5 * ((RHO_HIGH + RHO_LOW) -
							droplet[l] * (RHO_HIGH - RHO_LOW) * 
							tanh((radius - droplet_radius[l]) / INTERFACE_WIDTH * 2.0)));
						if (tmp > _A3(rho, k,j,i))
							_A3(rho, k,j,i) = tmp;
					}
				}

				// distribution function
				_A3(ux, k,j,i) = (real) 0.0;
				_A3(uy, k,j,i) = (real) 0.0;
				_A3(uz, k,j,i) = (real) 0.0;
				for (int q = 0; q < 19; ++q)
					_A4(fp, k,j,i,q) = _A4(fn, k,j,i,q) = wi[q] * _A3(rho, k,j,i);
			}
		}
	}
}

#endif // GENLBM_H

