// Variant Monte Carlo (VMC) solver for the disordered phi^4 depinning model.
//
// Model (arXiv:2306.13415):
//   dphi/dt = c*Laplacian(phi) + eps0*[(1+r(x,y))*phi - phi^3] + h
// with r(x,y) uncorrelated random-bond disorder, uniform in [-Delta,Delta]
// (Delta == AMPDIS), on an LxL grid, periodic in x and anti-periodic in y
// (so a single domain wall spans the sample). Elementary VMC move: replace
// phi_ij by the root of the cubic steady-state equation (dphi_ij/dt=0)
// closest to its current value, sweeping the lattice with a checkerboard
// decomposition (two-color updates run in parallel on the GPU).
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/swap.h>
#include <thrust/iterator/zip_iterator.h>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

typedef float REAL;

// model parameters (paper sets eps0=c=1 without loss of generality)
#ifndef AMPDIS
#define AMPDIS	0.2	// disorder strength Delta
#endif
#ifndef CEL
#define CEL	1.0	// elastic constant c
#endif
#ifndef EPSILON0
#define EPSILON0	1.0	// eps0
#endif
#ifndef R0
#define R0	1.0	// bare (undisordered) coefficient of phi
#endif

#ifndef TOLVEL
#define TOLVEL 0.0001	// convergence cutoff epsilon on the mean DW velocity
#endif

// safety cap on find_next_metastable's relaxation, in sweeps:
// max(MAXSWEEPS, MAXSWEEPS_PER_L*Lx). A relaxation that hasn't converged
// by then is reported as such (converged()==false), and the bisection
// counts it as depinned. Near h_d relaxations get very long at large L:
// at L=8192 a pinned trial took 8.3e5 sweeps to converge, and a trial
// that hit a 5e5 cap (then counted as pinned) escaped 2.3e5 sweeps
// later. Counting a capped trial as depinned keeps every saved critical
// configuration a confirmed fixed point; h_d can then only come out
// low, and only if a pinned relaxation outlasts the cap. (The original
// cap, 1e8 sweeps, was in practice an unbounded wait.)
#ifndef MAXSWEEPS
#define MAXSWEEPS 2000000
#endif
#ifndef MAXSWEEPS_PER_L
#define MAXSWEEPS_PER_L 500
#endif

// convergence cutoff on the largest residual max_i |dphi_i/dt| (local
// criterion, see find_next_metastable); 0 selects the legacy TOLVEL
// criterion on the mean DW velocity
#ifndef TOLRES
#define TOLRES 1e-5
#endif
// ... and, to tell a fixed point from the bottleneck of the saddle-node
// just above h_d (where the wall lingers almost still before escaping),
// no site may move more than CONFIRM_DPHI over CONFIRM_SWEEPS sweeps
#ifndef CONFIRM_SWEEPS
#define CONFIRM_SWEEPS 1000
#endif
#ifndef CONFIRM_DPHI
#define CONFIRM_DPHI 1e-4
#endif

/* counter-based random numbers, used both for the quenched disorder
   r(x,y) and to give every lattice site an independent, reproducible
   pseudo-random stream: http://www.thesalmons.org/john/random123/ */
#include <Random123/philox.h>
#include <Random123/u01.h>
typedef r123::Philox2x32 RNG;

// solves phi^3 + p*phi - q == 0 (Vieta/trigonometric method for the
// depressed cubic); returns 1 root (x1) or 3 roots (x1,x2,x3)
__device__ __host__
int vieta_solver(float p, float q, float &x1, float &x2, float &x3)
{
	float Q=p/3.;
	float R=q/2.;
	float D=Q*Q*Q+R*R;

	if(D>=0.0){
		x1=cbrtf(R+sqrtf(D))+cbrtf(R-sqrtf(D));
		return 1;
	}
	else{
		float theta=acosf(R/sqrtf(-Q*Q*Q));
		float two_sqrt_minus_Q=2*sqrtf(-Q);
		x1=two_sqrt_minus_Q*cosf(theta/3.0);
		x2=two_sqrt_minus_Q*cosf((theta+2.*M_PI)/3.0);
		x3=two_sqrt_minus_Q*cosf((theta+4.*M_PI)/3.0);
		return 3;
	}
}

// quenched random-bond disorder r(i,j), uniform in [-AMPDIS,AMPDIS],
// generated on the fly (no stored array) from a counter-based RNG keyed
// by the site index and the disorder seed, so it is deterministic and
// identical every time the same site is queried.
__device__
float site_disorder(int site_index, unsigned long disorder_seed)
{
	RNG rng;
	RNG::ctr_type c={{}};
	RNG::key_type k={{}};
	c[1]=uint32_t(disorder_seed);
	c[0]=uint32_t(site_index);
	k[0]=uint32_t(site_index);
	RNG::ctr_type r = rng(c, k);
	return AMPDIS*(2.0*u01_open_closed_32_53(r[0])-1.0);
}

// residual "velocity" dphi_ij/dt of the steady-state equation, used only
// to measure how close the lattice is to a metastable state (the VMC
// update itself does not use this functor)
struct velocityop
{
	int Lx, Ly;
	float * phi;
	float external_field;
	unsigned long disorder_seed;

	velocityop(float *_phi, int _Lx, int _Ly, float _external_field, unsigned long _disorder_seed):
	phi(_phi),Lx(_Lx),Ly(_Ly),external_field(_external_field),disorder_seed(_disorder_seed){};

	__device__
	float operator()(int i)
	{
		int x=i%Lx;
		int y=int(i/Lx);

		int y_plus1=(y+1)%Ly, y_minus1=(y-1+Ly)%Ly;
		int x_plus1=(x+1)%Lx, x_minus1=(x-1+Lx)%Lx;

		int up=x+Lx*y_minus1, down=x+Lx*y_plus1;
		int right=x_plus1+Lx*y, left=x_minus1+Lx*y;
		int center=x+Lx*y;

		float phi_up=phi[up], phi_down=phi[down];
		float phi_right=phi[right], phi_left=phi[left];
		float phi_center=phi[center];

		// anti-periodic BC in y, phi(x,y+Ly)=-phi(x,y): a single domain
		// wall spans the sample. The neighbor across the seam is "up"
		// (row Ly-1) for y=0 and "down" (row 0) for y=Ly-1.
		if(y==0) phi_up*=-1;
		if(y==Ly-1) phi_down*=-1;

		float laplacian = phi_up+phi_down+phi_right+phi_left-4.0*phi_center;
		float r = site_disorder(center, disorder_seed);
		float phi4_force = EPSILON0*(R0*(1.0+r)*phi_center - phi_center*phi_center*phi_center);

		return CEL*laplacian + phi4_force + external_field;
	}
};

// |dphi_ij/dt|
struct abs_velocityop
{
	velocityop v;
	abs_velocityop(velocityop _v):v(_v){};
	__device__
	float operator()(int i){ return fabsf(v(i)); }
};

// new value of a site with current value phi_center, neighbor sum
// sum_neighbors and disorder r: the root of the steady-state cubic
// closest to phi_center. Idempotent: with the same neighbors, it returns
// phi_center again if phi_center already is that root.
__device__
float vmc_site_update(float phi_center, float sum_neighbors, float r, float external_field)
{
	float r_center = R0*(1.0+r);

	/* Vieta form: phi^3 + p*phi - q == 0, derived from
	   0 = c*Laplacian(phi) + eps0*[(1+r)*phi - phi^3] + h
	   -> p = 4*c/eps0 - r_center ; q = (c*sum_neighbors + h)/eps0 */
	float p = 4.0*CEL/EPSILON0 - r_center;
	float q = (CEL*sum_neighbors + external_field)/EPSILON0;

	float x1,x2,x3;
	int num_roots = vieta_solver(p, q, x1, x2, x3);

	if(num_roots==1) return x1;
	// pick the root closest to the current value
	if(phi_center>x3) return x1;
	return x2;
}

// elementary VMC move: sets phi_ij to the root of the steady-state cubic
// closest to its current value. Sites are dispatched in a checkerboard
// (2-coloring) pattern so that a whole color can be updated in parallel:
// each site's four neighbors always belong to the other color.
struct vmcop
{
	int Lx, Ly;
	float * phi;
	float external_field;
	int color_offset;
	unsigned long disorder_seed;

	vmcop(float *_phi, int _Lx, int _Ly, float _external_field, int _color_offset, unsigned long _disorder_seed)
	:phi(_phi),Lx(_Lx),Ly(_Ly),external_field(_external_field),color_offset(_color_offset),disorder_seed(_disorder_seed)
	{};

	__device__
	void operator()(int j)
	{
		int i = (2*j) + (int((2*j)/Lx)+color_offset)%2;

		int x=i%Lx;
		int y=int(i/Lx);

		int y_plus1=(y+1)%Ly, y_minus1=(y-1+Ly)%Ly;
		int x_plus1=(x+1)%Lx, x_minus1=(x-1+Lx)%Lx;

		int up=x+Lx*y_minus1, down=x+Lx*y_plus1;
		int right=x_plus1+Lx*y, left=x_minus1+Lx*y;
		int center=x+Lx*y;

		float phi_up=phi[up], phi_down=phi[down];
		float phi_right=phi[right], phi_left=phi[left];
		float phi_center=phi[center];

		// anti-periodic BC in y, phi(x,y+Ly)=-phi(x,y): a single domain
		// wall spans the sample. The neighbor across the seam is "up"
		// (row Ly-1) for y=0 and "down" (row 0) for y=Ly-1.
		if(y==0) phi_up*=-1;
		if(y==Ly-1) phi_down*=-1;

		phi[center]=vmc_site_update(phi_center,phi_up+phi_down+phi_right+phi_left,
			site_disorder(center, disorder_seed),external_field);
	}
};

// Active tiles: the same checkerboard update, by TILE x TILE tiles, one
// thread block per tile, skipping inactive tiles. Since the update is
// idempotent, a site needs updating only if one of its neighbors changed
// in the previous (other-color) pass; a tile is active for a pass iff
// some site in it or in one of its 4 neighbor tiles changed in the
// previous pass. With tile_eps=0, skipping the others is exact (the
// result is bit-for-bit that of updating every site), but in float
// converged sites keep flickering in the last bit, so no tile ever goes
// quiet. A change |dphi|<=tile_eps is therefore applied but doesn't
// count as one; it leaves residuals ~ c*tile_eps on the neighbors, well
// below tolres, and every dynamics() call starts with a full sweep.
// work counts the active tiles' updates.
#define TILE 32
#ifndef TILE_EPS
#define TILE_EPS 1e-6
#endif
__global__
void vmc_tiles_kernel(float *phi, int Lx, int Ly, float external_field, int color_offset,
	unsigned long disorder_seed, const unsigned char *active, unsigned char *changed,
	unsigned long long *work, float tile_eps)
{
	int Tx=Lx/TILE;
	int tile=blockIdx.x;
	if(!active[tile]) return;
	if(threadIdx.x==0) atomicAdd(work,1ULL);

	int y=(tile/Tx)*TILE+threadIdx.x/(TILE/2);
	int x=(tile%Tx)*TILE+2*(threadIdx.x%(TILE/2));
	x+=(y+color_offset)&1; // this color: x = y+color_offset (mod 2), as in vmcop

	int y_plus1=(y+1)%Ly, y_minus1=(y-1+Ly)%Ly;
	int x_plus1=(x+1)%Lx, x_minus1=(x-1+Lx)%Lx;
	int center=x+Lx*y;
	float phi_up=phi[x+Lx*y_minus1], phi_down=phi[x+Lx*y_plus1];
	float phi_right=phi[x_plus1+Lx*y], phi_left=phi[x_minus1+Lx*y];
	float phi_center=phi[center];

	// anti-periodic BC in y (see velocityop)
	if(y==0) phi_up*=-1;
	if(y==Ly-1) phi_down*=-1;

	float new_phi=vmc_site_update(phi_center,phi_up+phi_down+phi_right+phi_left,
		site_disorder(center, disorder_seed),external_field);
	if(new_phi!=phi_center){
		phi[center]=new_phi;
		if(fabsf(new_phi-phi_center)>tile_eps) changed[tile]=1;
	}
}

// active[t] = some site changed in tile t or in one of its 4 neighbors
__global__
void spread_activity(const unsigned char *changed, unsigned char *active, int Tx, int Ty)
{
	int tile=blockIdx.x*blockDim.x+threadIdx.x;
	if(tile>=Tx*Ty) return;
	int tx=tile%Tx, ty=tile/Tx;
	active[tile]=changed[tile] | changed[(tx+1)%Tx+Tx*ty] | changed[(tx-1+Tx)%Tx+Tx*ty]
		| changed[tx+Tx*((ty+1)%Ty)] | changed[tx+Tx*((ty-1+Ty)%Ty)];
}

__host__ __device__ int index_center(int x,int y,int Lx,int Ly){ return x+Lx*y; }
__host__ __device__ int index_up(int x,int y,int Lx,int Ly){ return x+Lx*((y-1+Ly)%Ly); }
__host__ __device__ int index_down(int x,int y,int Lx,int Ly){ return x+Lx*((y+1)%Ly); }
__host__ __device__ int index_right(int x,int y,int Lx,int Ly){ return (x+1)%Lx+Lx*y; }
__host__ __device__ int index_left(int x,int y,int Lx,int Ly){ return (x-1+Lx)%Lx+Lx*y; }

// detects domain-wall sites, restricted to the [x1,x2]x[y1,y2] window
// (to avoid picking up the periodic-image seam). Each site is first
// replaced by its 5-point (self+4 neighbors) smoothed value, and a wall
// site is one where the smoothed value changes sign between the row
// above and the row below -- this smoothing is what lets a single,
// well-defined interface height be extracted even where the raw
// configuration has overhangs/pinch-off loops (see paper, Fig. 1).
struct is_wall_smooth
{
	int Lx, Ly;
	float *phi;
	int x1, x2, y1, y2;
	is_wall_smooth(float *_phi, int _Lx, int _Ly, int _x1, int _x2, int _y1, int _y2):
	phi(_phi),Lx(_Lx),Ly(_Ly),x1(_x1),x2(_x2),y1(_y1),y2(_y2){};

	__device__
	float smoothed(int x, int y)
	{
		return phi[index_center(x,y,Lx,Ly)]+phi[index_up(x,y,Lx,Ly)]+phi[index_down(x,y,Lx,Ly)]
			+phi[index_right(x,y,Lx,Ly)]+phi[index_left(x,y,Lx,Ly)];
	}

	__device__
	bool operator()(int i)
	{
		int x=i%Lx;
		int y=int(i/Lx);

		float smoothed_up=smoothed(x,(y-1+Ly)%Ly);
		float smoothed_down=smoothed(x,(y+1)%Ly);

		bool sign_change = (smoothed_up*smoothed_down<0);
		return sign_change && x>x1 && x<x2 && y>y1 && y<y2;
	}
};

// |a_i-b_i|, reduced with max to compare two fields
struct abs_diff
{
	__host__ __device__
	float operator()(const thrust::tuple<float,float> &t){ return fabsf(thrust::get<0>(t)-thrust::get<1>(t)); }
};

// row of site i if phi_i has the given sign, else a neutral value
struct row_if_sign
{
	float *phi;
	int Lx, sign, neutral;
	row_if_sign(float *_phi, int _Lx, int _sign, int _neutral):phi(_phi),Lx(_Lx),sign(_sign),neutral(_neutral){};
	__device__
	int operator()(int i){ return phi[i]*sign>0 ? i/Lx : neutral; }
};

struct is_positive
{
	__host__ __device__
	bool operator()(float x){ return x > 0; }
};

class System{
	private:
		thrust::device_ptr<float> d_phi;
		thrust::device_ptr<float> d_saved; // a copy of phi (save_state/restore_state)
		thrust::device_ptr<float> d_snap;  // phi some sweeps ago (find_next_metastable)
		thrust::device_vector<int> d_wall_site_indices; // wall-site indices (detect_wall), sized to fit
		// active tiles (see vmc_tiles_kernel)
		bool use_tiles;
		float tile_eps;
		thrust::device_vector<unsigned char> d_active, d_changed;
		thrust::device_vector<unsigned long long> d_work;
		int wall_first_index;
		int wall_last_index;

		int Lx;   // along the wall (periodic)
		int Ly;   // along the motion (anti-periodic)
		int L2;   // Lx*Ly

		float external_field;
		int total_time;
		float tolres;
		bool last_converged; // did the last find_next_metastable converge?
		long max_sweeps;     // its cap
		// geometry of a trial: the wall starts flat at row y_start and has
		// escaped once it crosses row escape_row: some site of the row
		// (escape_full=false, the legacy criterion) or every site of it
		// (escape_full=true; a pinned wall may then stick out above
		// escape_row, into the headroom below Ly)
		int y_start, escape_row;
		bool escape_full;

	public:
		unsigned long disorder_seed;
		float *phi_ptr;

		System(int _Lx, int _Ly, unsigned long _disorder_seed)
		{
			if(_Lx%2!=0 || _Ly%2!=0){
				std::cout << "ERROR: Lx and Ly must be even for checkerboard VMC" << std::endl;
				exit(1);
			}

			Lx=_Lx;
			Ly=_Ly;
			L2=Lx*Ly;

			d_phi = thrust::device_malloc<float>(L2);
			d_saved = thrust::device_malloc<float>(L2);
			d_snap = thrust::device_malloc<float>(L2);
			phi_ptr = thrust::raw_pointer_cast(&d_phi[0]);

			use_tiles = (Lx%TILE==0 && Ly%TILE==0);
			tile_eps = TILE_EPS;
			if(use_tiles){
				d_active.resize((Lx/TILE)*(Ly/TILE));
				d_changed.resize((Lx/TILE)*(Ly/TILE));
				d_work.resize(1,0);
			}

			external_field=0.0;
			total_time=0;
			tolres=TOLRES;
			last_converged=true;
			max_sweeps=std::max((long)MAXSWEEPS,(long)MAXSWEEPS_PER_L*Lx);
			y_start=int(Ly*0.2);
			escape_row=int(0.95f*Ly);
			escape_full=false;
			disorder_seed=_disorder_seed;

			reset_uniform();

			std::ofstream log("logfile.dat");
			log << "L= " << Lx << std::endl;
			log << "Ly= " << Ly << std::endl;
			log << "disorder_seed= " << disorder_seed << std::endl;
			log << "AMPDIS (Delta)= " << AMPDIS << std::endl;
			log << "CEL (c)= " << CEL << std::endl;
			log << "EPSILON0= " << EPSILON0 << std::endl;
			log << "R0= " << R0 << std::endl;
			log << "TOLVEL (epsilon)= " << TOLVEL << std::endl;
		}

		void set_tolres(float t){ tolres=t; }
		float get_tolres(){ return tolres; }

		~System(){
			thrust::device_free(d_phi);
			thrust::device_free(d_saved);
			thrust::device_free(d_snap);
		}

		// flat interface: phi=+1 for y<y_level, phi=-1 for y>=y_level
		void set_flat_interface(int y_level)
		{
			int split=y_level*Lx;
			thrust::fill(d_phi, d_phi+split, 1);
			thrust::fill(d_phi+split, d_phi+L2, -1);
		};
		void set_flat_interface(){ set_flat_interface(y_start); }

		void set_geometry(int _y_start, int _escape_row, bool _escape_full)
		{
			y_start=_y_start; escape_row=_escape_row; escape_full=_escape_full;
		}
		int get_y_start(){ return y_start; }
		int get_escape_row(){ return escape_row; }
		bool get_escape_full(){ return escape_full; }

		void reset_uniform(){ thrust::fill(d_phi, d_phi+L2, -1.0f); };

		// keep a copy of phi / go back to it (warm starts of the bisection)
		void save_state(){ thrust::copy(d_phi, d_phi+L2, d_saved); }
		void restore_state(){ thrust::copy(d_saved, d_saved+L2, d_phi); }
		void swap_state(){ thrust::swap_ranges(d_phi, d_phi+L2, d_saved); }

		// max_i |phi_i - other_i|
		float max_diff(thrust::device_ptr<float> other)
		{
			return thrust::transform_reduce(
				thrust::make_zip_iterator(thrust::make_tuple(d_phi, other)),
				thrust::make_zip_iterator(thrust::make_tuple(d_phi+L2, other+L2)),
				abs_diff(), 0.0f, thrust::maximum<float>());
		}
		float max_diff_saved(){ return max_diff(d_saved); }

		void set_external_field(float h){ external_field=h; }
		float get_external_field(){ return external_field; }
		int get_L(){ return Lx; }
		int get_Ly(){ return Ly; }
		int get_total_time(){ return total_time; }

		// whole-lattice sweeps instead of active tiles (for comparisons)
		void set_use_tiles(bool t){ use_tiles = t && (Lx%TILE==0 && Ly%TILE==0); }
		bool get_use_tiles(){ return use_tiles; }
		void set_tile_eps(float e){ tile_eps=e; }

		// sweeps' worth of tile updates done so far (one sweep = 2*(Lx/TILE)*(Ly/TILE))
		double tile_work(){
			if(!use_tiles) return 0;
			unsigned long long w=d_work[0];
			return double(w)/(2.0*(Lx/TILE)*(Ly/TILE));
		}

		// checkerboard VMC sweeps: num_sweeps sweeps of the lattice
		void dynamics(int num_sweeps)
		{
			if(use_tiles){ dynamics_tiles(num_sweeps); return; }
			int half_L2=L2/2;
			for(int sweep=0;sweep<num_sweeps;sweep++)
			{
				for(int color_offset=0;color_offset<2;color_offset++){
					thrust::for_each(
						thrust::make_counting_iterator(0),
						thrust::make_counting_iterator(half_L2),
						vmcop(phi_ptr,Lx,Ly,external_field,color_offset,disorder_seed)
					);
				}
				total_time++;
			}
		}

		// the same sweeps, by active tiles. The first sweep of every call
		// (both colors) updates all tiles, since phi or h may have been
		// changed in between.
		void dynamics_tiles(int num_sweeps)
		{
			int Tx=Lx/TILE, Ty=Ly/TILE, ntiles=Tx*Ty;
			unsigned char *active=thrust::raw_pointer_cast(d_active.data());
			unsigned char *changed=thrust::raw_pointer_cast(d_changed.data());
			unsigned long long *work=thrust::raw_pointer_cast(d_work.data());
			cudaMemset(active,1,ntiles);
			for(int sweep=0;sweep<num_sweeps;sweep++)
			{
				for(int color_offset=0;color_offset<2;color_offset++){
					cudaMemset(changed,0,ntiles);
					vmc_tiles_kernel<<<ntiles,TILE*TILE/2>>>(phi_ptr,Lx,Ly,external_field,color_offset,
						disorder_seed,active,changed,work,tile_eps);
					if(sweep==0 && color_offset==0) cudaMemset(active,1,ntiles);
					else spread_activity<<<(ntiles+255)/256,256>>>(changed,active,Tx,Ty);
				}
				total_time++;
			}
		}

		// mean residual velocity <dphi/dt>, i.e. the paper's convergence
		// diagnostic (note: normalized by Lx, not L2, matching the scale
		// at which TOLVEL was tuned)
		float velocity_phi()
		{
			return thrust::transform_reduce(
				thrust::make_counting_iterator(0),
				thrust::make_counting_iterator(L2),
				velocityop(phi_ptr,Lx,Ly,external_field,disorder_seed),
				float(0.0), thrust::plus<float>()
			)/float(Lx);
		}

		// largest residual max_i |dphi_i/dt|: zero only at a genuine fixed
		// point of the VMC map
		float max_residual()
		{
			return thrust::transform_reduce(
				thrust::make_counting_iterator(0),
				thrust::make_counting_iterator(L2),
				abs_velocityop(velocityop(phi_ptr,Lx,Ly,external_field,disorder_seed)),
				0.0f, thrust::maximum<float>());
		}

		// runs elementary chunks of "chop" VMC sweeps until a metastable
		// state has been reached: max_i |dphi_i/dt| < tolres. (With
		// tolres=0, the legacy criterion: mean DW velocity < TOLVEL. That
		// one is blind to a local avalanche of size l, which only adds
		// ~l*v/Lx to the mean, so it can stop on a still-moving wall.)
		// Once max|dphi/dt| < tolres, it is confirmed by checking that no
		// site moves more than CONFIRM_DPHI over the next CONFIRM_SWEEPS
		// sweeps, which a wall in the saddle-node bottleneck still does.
		// If check_escape, also stops as soon as the interface has
		// escaped (escaped()): for bisection that already decides
		// "depinned", so there's no need to relax the rest of the way.
		// Gives up after max_sweeps sweeps. Returns the number of sweeps
		// done; converged() tells whether it stopped at a fixed point.
		int find_next_metastable(int chop, bool check_escape=false)
		{
			bool moving;
			int iterations=0;
			int snap_iteration=-1; // iteration of the snapshot, -1: none
			do{
				dynamics(chop);
				iterations++;
				if(tolres>0){
					moving = max_residual()>tolres;
					if(moving) snap_iteration=-1;
					else if(snap_iteration<0){
						thrust::copy(d_phi, d_phi+L2, d_snap);
						snap_iteration=iterations;
						moving=true;
					}
					else if((iterations-snap_iteration)*chop<CONFIRM_SWEEPS) moving=true;
					else if(max_diff(d_snap)>CONFIRM_DPHI){
						thrust::copy(d_phi, d_phi+L2, d_snap);
						snap_iteration=iterations;
						moving=true;
					}
				}
				else moving = fabs(velocity_phi())>TOLVEL;
				if(check_escape && escaped()) break;
			}while(moving && (long)iterations*chop<max_sweeps);

			last_converged=!moving;
			return iterations*chop;
		}
		bool converged(){ return last_converged; }

		float magnetization(){ return thrust::reduce(d_phi,d_phi+L2); }

		// max / min of phi along row y: max>0 means the wall has reached
		// that row somewhere, min>0 that all of it has swept past
		float row_max(int y){ return thrust::reduce(d_phi+Lx*y,d_phi+Lx*(y+1),float(-10.),thrust::maximum<float>()); }
		float row_min(int y){ return thrust::reduce(d_phi+Lx*y,d_phi+Lx*(y+1),float(10.),thrust::minimum<float>()); }

		// has the interface escaped (see escape_row)?
		bool escaped(){ return escape_full ? row_min(escape_row)>0 : row_max(escape_row)>0; }

		// rows spanned by the wall: (lowest row with some phi<0, highest
		// row with some phi>0); the first also catches phi<0 bubbles left
		// behind, the second is the top of the wall
		thrust::pair<int,int> wall_rows()
		{
			int bottom=thrust::transform_reduce(thrust::make_counting_iterator(0),thrust::make_counting_iterator(L2),
				row_if_sign(phi_ptr,Lx,-1,Ly),Ly,thrust::minimum<int>());
			int top=thrust::transform_reduce(thrust::make_counting_iterator(0),thrust::make_counting_iterator(L2),
				row_if_sign(phi_ptr,Lx,+1,-1),-1,thrust::maximum<int>());
			return thrust::make_pair(bottom,top);
		}

		float local_magnetization(int x, int y){ return d_phi[(x+Lx*y)]; }

		thrust::tuple<int,int,int,int,int> detect_wall(int x1, int x2, int y1, int y2)
		{
			// counted first, so the buffer is only as large as the wall
			// (rather than a whole extra field's worth of GPU memory)
			is_wall_smooth is_wall(phi_ptr,Lx,Ly,x1,x2,y1,y2);
			int num_wall_points = thrust::count_if(
				thrust::make_counting_iterator(0), thrust::make_counting_iterator(L2), is_wall);
			d_wall_site_indices.resize(num_wall_points);
			thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(L2),
				d_wall_site_indices.begin(), is_wall);

			int area_positive = thrust::count_if(thrust::device, phi_ptr, phi_ptr+L2, is_positive());
			int area_negative = L2 - area_positive;

			thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> result =
				thrust::minmax_element(d_wall_site_indices.begin(), d_wall_site_indices.end());

			wall_last_index = num_wall_points ? int(*result.second) : -1;
			wall_first_index = num_wall_points ? int(*result.first) : -1;

			int last_y=int(wall_last_index/Lx);
			int first_y=int(wall_first_index/Lx);

			return thrust::tuple<int,int,int,int,int>(num_wall_points,area_positive,area_negative,first_y,last_y);
		}

		// saves the domain-wall coordinates (the "critical configuration"
		// once called at h=h_d) restricted to x in (x1,x2), y in (y1,y2)
		int print_wall(std::ofstream &fout, int x1, int x2, int y1, int y2)
		{
			thrust::tuple<int,int,int,int,int> stats = detect_wall(x1, x2, y1, y2);
			int num_wall_points=thrust::get<0>(stats);
			int area_positive=thrust::get<1>(stats);
			int area_negative=thrust::get<2>(stats);

			std::cout << "# " << num_wall_points << " " << area_positive << " " << area_negative
				<< " " << wall_first_index << " " << wall_last_index << std::endl;
			fout << "# " << num_wall_points << " " << area_positive << " " << area_negative
				<< " " << wall_first_index << " " << wall_last_index << std::endl;

			thrust::host_vector<int> wall_points(d_wall_site_indices);
			d_wall_site_indices.clear();
			d_wall_site_indices.shrink_to_fit();

			for(int i=0;i<num_wall_points;i++){
				int x=wall_points[i]%Lx;
				int y=int(wall_points[i]/Lx);
				fout << x << " " << y << std::endl;
			}
			fout << std::endl << std::endl;

			return num_wall_points;
		}

		// per-column and per-row magnetization profile of the whole field:
		// line i holds column i's and row i's sums (nan past Lx or Ly).
		// Computed on the host, so it costs no GPU memory.
		void print_magnetization_profile(std::ofstream &fout)
		{
			thrust::host_vector<float> h_phi(d_phi,d_phi+L2);
			std::vector<double> mag_by_x(Lx,0.0), mag_by_y(Ly,0.0);
			for(int y=0;y<Ly;y++)
				for(int x=0;x<Lx;x++){
					float v=h_phi[x+(size_t)Lx*y];
					mag_by_x[x]+=v; mag_by_y[y]+=v;
				}
			for(int i=0;i<std::max(Lx,Ly);i++){
				if(i<Lx) fout << float(mag_by_x[i]); else fout << "nan";
				fout << " ";
				if(i<Ly) fout << float(mag_by_y[i]); else fout << "nan";
				fout << std::endl;
			}
			fout << "\n" << std::endl;
		}

		// raw float32 dump of the whole field, row-major (index x+y*Lx, Ly
		// rows of Lx), for
		// offline contour extraction (scripts/wall_contour.py); fout must be
		// opened in binary mode
		void print_field_binary(std::ofstream &fout)
		{
			thrust::host_vector<float> h_phi(d_phi,d_phi+L2);
			fout.write(reinterpret_cast<const char*>(h_phi.data()),sizeof(float)*L2);
		}

		// reads a field written by print_field_binary
		void read_field_binary(std::ifstream &fin)
		{
			thrust::host_vector<float> h_phi(L2);
			fin.read(reinterpret_cast<char*>(h_phi.data()),sizeof(float)*L2);
			thrust::copy(h_phi.begin(),h_phi.end(),d_phi);
		}

		void print_config(std::ofstream &fout)
		{
			for(int i=0;i<Ly;i++){
				for(int j=0;j<Lx-1;j++) fout << d_phi[j+i*Lx] << " ";
				fout << d_phi[Lx-1+i*Lx] << "\n";
			}
			fout << "\n" << std::endl;
		}

		void print_config_linear(std::ofstream &fout)
		{
			for(int i=0;i<L2;i++) fout << d_phi[i] << std::endl;
			fout << "\n" << std::endl;
		}

		void read_config_linear(std::ifstream &fin)
		{
			float x;
			for(int i=0;i<L2;i++){ fin >> x; d_phi[i]=x; }
		}
};
