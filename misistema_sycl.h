// SYCL port of misistema.h, for Intel GPUs (Data Center GPU Max / PVC);
// the CUDA version stays the reference. Same model, same algorithm, same
// disorder samples (site_disorder is bit-for-bit the same); Thrust is
// replaced by oneDPL and the CUDA kernels by SYCL kernels on an in-order
// queue, with USM device memory. Comments on the physics and on the
// algorithm are in misistema.h; here only what differs is explained.
//
// Model (arXiv:2306.13415):
//   dphi/dt = c*Laplacian(phi) + eps0*[(1+r(x,y))*phi - phi^3] + h
// on an Lx x Ly grid, periodic (or helical) in x, anti-periodic in y.

// oneDPL's headers must come before any standard library header
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>
#include <sycl/sycl.hpp>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <tuple>
#include <utility>

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
// r(x,y) is constant over blocks of AX x AY sites (1x1: uncorrelated)
#ifndef AX
#define AX 1
#endif
#ifndef AY
#define AY 1
#endif

#ifndef TOLVEL
#define TOLVEL 0.0001	// convergence cutoff epsilon on the mean DW velocity
#endif

// safety cap on find_next_metastable's relaxation, in sweeps:
// max(MAXSWEEPS, MAXSWEEPS_PER_L*Lx) (see misistema.h)
#ifndef MAXSWEEPS
#define MAXSWEEPS 2000000
#endif
#ifndef MAXSWEEPS_PER_L
#define MAXSWEEPS_PER_L 2000
#endif

// convergence cutoff on max_i |dphi_i/dt| (0: legacy TOLVEL criterion),
// confirmed by no site moving more than CONFIRM_DPHI over CONFIRM_SWEEPS
#ifndef TOLRES
#define TOLRES 1e-5
#endif
#ifndef CONFIRM_SWEEPS
#define CONFIRM_SWEEPS 1000
#endif
#ifndef CONFIRM_DPHI
#define CONFIRM_DPHI 1e-4
#endif

/* counter-based random numbers: http://www.thesalmons.org/john/random123/ */
#include <Random123/philox.h>
#include <Random123/u01.h>
typedef r123::Philox2x32 RNG;

namespace dpl = oneapi::dpl;

// solves phi^3 + p*phi - q == 0 (Vieta/trigonometric method for the
// depressed cubic); returns 1 root (x1) or 3 roots (x1,x2,x3). Written
// exactly as in misistema.h (the C float math functions are available in
// SYCL device code).
inline
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

// quenched random-bond disorder r(x,y), uniform in [-AMPDIS,AMPDIS],
// generated on the fly from a counter-based RNG keyed by the site's block
// and the disorder seed (identical to misistema.h: same samples)
inline
float site_disorder(int x, int y, int Lx, unsigned long disorder_seed)
{
	int site_index=(x/AX)+(Lx/AX)*(y/AY);
	RNG rng;
	RNG::ctr_type c={{}};
	RNG::key_type k={{}};
	c[1]=uint32_t(disorder_seed);
	c[0]=uint32_t(site_index);
	k[0]=uint32_t(site_index);
	RNG::ctr_type r = rng(c, k);
	return AMPDIS*(2.0*u01_open_closed_32_53(r[0])-1.0);
}

// phi of the right (dir=+1) or left (dir=-1) neighbor of (x,y), with the
// helical BC in x and the anti-periodic seam in y (see misistema.h)
inline
float phi_x_neighbor(const float *phi, int x, int y, int Lx, int Ly, int tilt, int dir)
{
	int xn=x+dir, yn=y;
	if(xn==Lx){ xn=0; yn=y-tilt; }
	else if(xn<0){ xn=Lx-1; yn=y+tilt; }
	float sign=1;
	while(yn>=Ly){ yn-=Ly; sign=-sign; }
	while(yn<0){ yn+=Ly; sign=-sign; }
	return sign*phi[xn+Lx*yn];
}

// residual "velocity" dphi_ij/dt of the steady-state equation
struct velocityop
{
	int Lx, Ly, tilt;
	float * phi;
	float external_field;
	unsigned long disorder_seed;

	velocityop(float *_phi, int _Lx, int _Ly, int _tilt, float _external_field, unsigned long _disorder_seed):
	Lx(_Lx),Ly(_Ly),tilt(_tilt),phi(_phi),external_field(_external_field),disorder_seed(_disorder_seed){};

	float operator()(int i) const
	{
		int x=i%Lx;
		int y=int(i/Lx);

		int y_plus1=(y+1)%Ly, y_minus1=(y-1+Ly)%Ly;

		int up=x+Lx*y_minus1, down=x+Lx*y_plus1;
		int center=x+Lx*y;

		float phi_up=phi[up], phi_down=phi[down];
		float phi_right=phi_x_neighbor(phi,x,y,Lx,Ly,tilt,1), phi_left=phi_x_neighbor(phi,x,y,Lx,Ly,tilt,-1);
		float phi_center=phi[center];

		// anti-periodic BC in y
		if(y==0) phi_up*=-1;
		if(y==Ly-1) phi_down*=-1;

		float laplacian = phi_up+phi_down+phi_right+phi_left-4.0*phi_center;
		float r = site_disorder(x, y, Lx, disorder_seed);
		float phi4_force = EPSILON0*(R0*(1.0+r)*phi_center - phi_center*phi_center*phi_center);

		return CEL*laplacian + phi4_force + external_field;
	}
};

// |dphi_ij/dt|
struct abs_velocityop
{
	velocityop v;
	abs_velocityop(velocityop _v):v(_v){};
	float operator()(int i) const { return fabsf(v(i)); }
};

// new value of a site: the root of the steady-state cubic closest to
// phi_center (see misistema.h)
inline
float vmc_site_update(float phi_center, float sum_neighbors, float r, float external_field)
{
	float r_center = R0*(1.0+r);

	float p = 4.0*CEL/EPSILON0 - r_center;
	float q = (CEL*sum_neighbors + external_field)/EPSILON0;

	float x1,x2,x3;
	int num_roots = vieta_solver(p, q, x1, x2, x3);

	if(num_roots==1) return x1;
	// pick the root closest to the current value
	if(phi_center>x3) return x1;
	return x2;
}

// elementary VMC move on site j of one checkerboard color
struct vmcop
{
	int Lx, Ly, tilt;
	float * phi;
	float external_field;
	int color_offset;
	unsigned long disorder_seed;

	vmcop(float *_phi, int _Lx, int _Ly, int _tilt, float _external_field, int _color_offset, unsigned long _disorder_seed)
	:Lx(_Lx),Ly(_Ly),tilt(_tilt),phi(_phi),external_field(_external_field),color_offset(_color_offset),disorder_seed(_disorder_seed)
	{};

	void operator()(int j) const
	{
		int i = (2*j) + (int((2*j)/Lx)+color_offset)%2;

		int x=i%Lx;
		int y=int(i/Lx);

		int y_plus1=(y+1)%Ly, y_minus1=(y-1+Ly)%Ly;

		int up=x+Lx*y_minus1, down=x+Lx*y_plus1;
		int center=x+Lx*y;

		float phi_up=phi[up], phi_down=phi[down];
		float phi_right=phi_x_neighbor(phi,x,y,Lx,Ly,tilt,1), phi_left=phi_x_neighbor(phi,x,y,Lx,Ly,tilt,-1);
		float phi_center=phi[center];

		// anti-periodic BC in y
		if(y==0) phi_up*=-1;
		if(y==Ly-1) phi_down*=-1;

		phi[center]=vmc_site_update(phi_center,phi_up+phi_down+phi_right+phi_left,
			site_disorder(x, y, Lx, disorder_seed),external_field);
	}
};

// Active tiles (see misistema.h): one work-group of TILE*TILE/2 work-items
// per TILE x TILE tile, skipping inactive tiles
#define TILE 32
#ifndef TILE_EPS
#define TILE_EPS 1e-6
#endif
inline
void vmc_tiles_kernel(sycl::nd_item<1> item, float *phi, int Lx, int Ly, int tilt, float external_field, int color_offset,
	unsigned long disorder_seed, const unsigned char *active, unsigned char *changed,
	unsigned long long *work, float tile_eps)
{
	int Tx=Lx/TILE;
	int tile=item.get_group(0);
	int thread=item.get_local_id(0);
	if(!active[tile]) return;
	if(thread==0){
		sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
			sycl::access::address_space::global_space> w(*work);
		w.fetch_add(1ULL);
	}

	int y=(tile/Tx)*TILE+thread/(TILE/2);
	int x=(tile%Tx)*TILE+2*(thread%(TILE/2));
	x+=(y+color_offset)&1; // this color: x = y+color_offset (mod 2), as in vmcop

	int y_plus1=(y+1)%Ly, y_minus1=(y-1+Ly)%Ly;
	int center=x+Lx*y;
	float phi_up=phi[x+Lx*y_minus1], phi_down=phi[x+Lx*y_plus1];
	float phi_right=phi_x_neighbor(phi,x,y,Lx,Ly,tilt,1), phi_left=phi_x_neighbor(phi,x,y,Lx,Ly,tilt,-1);
	float phi_center=phi[center];

	// anti-periodic BC in y
	if(y==0) phi_up*=-1;
	if(y==Ly-1) phi_down*=-1;

	float new_phi=vmc_site_update(phi_center,phi_up+phi_down+phi_right+phi_left,
		site_disorder(x, y, Lx, disorder_seed),external_field);
	if(new_phi!=phi_center){
		phi[center]=new_phi;
		if(fabsf(new_phi-phi_center)>tile_eps) changed[tile]=1;
	}
}

// active[t] = some site changed in tile t or in one of its 4 neighbors
// (across the x boundary, the tiles the helical BC connects to)
inline
unsigned char changed_across(const unsigned char *changed, int tx_other, int ty, int Tx, int Ty, int shift)
{
	int Ly=Ty*TILE;
	int y0=((ty*TILE+shift)%Ly+Ly)%Ly, y1=((ty*TILE+TILE-1+shift)%Ly+Ly)%Ly;
	return changed[tx_other+Tx*(y0/TILE)] | changed[tx_other+Tx*(y1/TILE)];
}
inline
void spread_activity(int tile, const unsigned char *changed, unsigned char *active, int Tx, int Ty, int tilt)
{
	int tx=tile%Tx, ty=tile/Tx;
	unsigned char right = tx==Tx-1 ? changed_across(changed,0,ty,Tx,Ty,-tilt) : changed[tx+1+Tx*ty];
	unsigned char left = tx==0 ? changed_across(changed,Tx-1,ty,Tx,Ty,tilt) : changed[tx-1+Tx*ty];
	active[tile]=changed[tile] | right | left
		| changed[tx+Tx*((ty+1)%Ty)] | changed[tx+Tx*((ty-1+Ty)%Ty)];
}

inline int index_center(int x,int y,int Lx,int Ly){ return x+Lx*y; }
inline int index_up(int x,int y,int Lx,int Ly){ return x+Lx*((y-1+Ly)%Ly); }
inline int index_down(int x,int y,int Lx,int Ly){ return x+Lx*((y+1)%Ly); }

// detects domain-wall sites in the [x1,x2]x[y1,y2] window, from the
// 5-point smoothed field (see misistema.h)
struct is_wall_smooth
{
	int Lx, Ly, tilt;
	float *phi;
	int x1, x2, y1, y2;
	is_wall_smooth(float *_phi, int _Lx, int _Ly, int _tilt, int _x1, int _x2, int _y1, int _y2):
	Lx(_Lx),Ly(_Ly),tilt(_tilt),phi(_phi),x1(_x1),x2(_x2),y1(_y1),y2(_y2){};

	float smoothed(int x, int y) const
	{
		return phi[index_center(x,y,Lx,Ly)]+phi[index_up(x,y,Lx,Ly)]+phi[index_down(x,y,Lx,Ly)]
			+phi_x_neighbor(phi,x,y,Lx,Ly,tilt,1)+phi_x_neighbor(phi,x,y,Lx,Ly,tilt,-1);
	}

	bool operator()(int i) const
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
	float operator()(float a, float b) const { return fabsf(a-b); }
};

// row of site i if phi_i has the given sign, else a neutral value
struct row_if_sign
{
	float *phi;
	int Lx, sign, neutral;
	row_if_sign(float *_phi, int _Lx, int _sign, int _neutral):phi(_phi),Lx(_Lx),sign(_sign),neutral(_neutral){};
	int operator()(int i) const { return phi[i]*sign>0 ? i/Lx : neutral; }
};

struct is_positive
{
	bool operator()(float x) const { return x > 0; }
};

// initial straight wall y = y_low + tilt*x/Lx (phi=+1 below, -1 above)
struct tilted_wall
{
	int Lx, y_low, tilt;
	tilted_wall(int _Lx, int _y_low, int _tilt):Lx(_Lx),y_low(_y_low),tilt(_tilt){};
	float operator()(int i) const {
		int x=i%Lx, y=i/Lx;
		return y < y_low + (long(tilt)*x)/Lx ? 1.0f : -1.0f;
	}
};

class System{
	private:
		// in-order queue: kernels and copies run in submission order.
		// The device is SYCL's default one, so ONEAPI_DEVICE_SELECTOR (and
		// ZE_AFFINITY_MASK, to pick a PVC stack) choose it.
		sycl::queue q;
		// oneDPL policy on q; its algorithms are blocking
		decltype(dpl::execution::make_device_policy(std::declval<sycl::queue&>())) policy;

		float *d_phi;
		float *d_saved; // a copy of phi (save_state/restore_state)
		float *d_snap;  // phi some sweeps ago (find_next_metastable)
		int *d_wall_site_indices; // wall-site indices (detect_wall), sized to fit
		int num_wall_site_indices;
		// active tiles (see vmc_tiles_kernel)
		bool use_tiles;
		float tile_eps;
		unsigned char *d_active, *d_changed;
		unsigned long long *d_work;
		int wall_first_index;
		int wall_last_index;

		int Lx;   // along the wall (periodic, or helical with tilt)
		int Ly;   // along the motion (anti-periodic)
		int tilt; // helical BC phi(x+Lx,y)=phi(x,y-tilt): mean wall slope tilt/Lx
		int L2;   // Lx*Ly

		float external_field;
		int total_time;
		float tolres;
		bool last_converged; // did the last find_next_metastable converge?
		long max_sweeps;     // its cap
		// geometry of a trial (see misistema.h)
		int y_start, escape_row;
		bool escape_full;

		dpl::counting_iterator<int> site(int i){ return dpl::counting_iterator<int>(i); }

	public:
		unsigned long disorder_seed;
		float *phi_ptr;

		System(int _Lx, int _Ly, unsigned long _disorder_seed)
		: q(sycl::default_selector_v, sycl::property::queue::in_order()),
		  policy(dpl::execution::make_device_policy(q))
		{
			if(_Lx%2!=0 || _Ly%2!=0){
				std::cout << "ERROR: Lx and Ly must be even for checkerboard VMC" << std::endl;
				exit(1);
			}

			if(_Lx%AX!=0){
				std::cout << "ERROR: Lx must be a multiple of AX=" << AX << std::endl;
				exit(1);
			}

			Lx=_Lx;
			Ly=_Ly;
			L2=Lx*Ly;
			tilt=0;

			d_phi = sycl::malloc_device<float>(L2, q);
			d_saved = sycl::malloc_device<float>(L2, q);
			d_snap = sycl::malloc_device<float>(L2, q);
			if(!d_phi || !d_saved || !d_snap){
				std::cerr << "ERROR: cannot allocate 3 x " << L2 << " floats on the device" << std::endl;
				exit(1);
			}
			phi_ptr = d_phi;
			d_wall_site_indices = nullptr;
			num_wall_site_indices = 0;

			use_tiles = (Lx%TILE==0 && Ly%TILE==0);
			tile_eps = TILE_EPS;
			int ntiles=(Lx/TILE)*(Ly/TILE);
			d_active = sycl::malloc_device<unsigned char>(std::max(ntiles,1), q);
			d_changed = sycl::malloc_device<unsigned char>(std::max(ntiles,1), q);
			d_work = sycl::malloc_device<unsigned long long>(1, q);
			q.memset(d_work, 0, sizeof(unsigned long long)).wait();

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

			std::string device=q.get_device().get_info<sycl::info::device::name>();
			std::cerr << "SYCL device: " << device << std::endl;

			std::ofstream log("logfile.dat");
			log << "L= " << Lx << std::endl;
			log << "Ly= " << Ly << std::endl;
			log << "disorder_seed= " << disorder_seed << std::endl;
			log << "AMPDIS (Delta)= " << AMPDIS << std::endl;
			log << "CEL (c)= " << CEL << std::endl;
			log << "EPSILON0= " << EPSILON0 << std::endl;
			log << "R0= " << R0 << std::endl;
			log << "AX= " << AX << std::endl;
			log << "AY= " << AY << std::endl;
			log << "TOLVEL (epsilon)= " << TOLVEL << std::endl;
			log << "device= " << device << std::endl;
		}

		void set_tolres(float t){ tolres=t; }
		float get_tolres(){ return tolres; }

		~System(){
			sycl::free(d_phi, q);
			sycl::free(d_saved, q);
			sycl::free(d_snap, q);
			sycl::free(d_active, q);
			sycl::free(d_changed, q);
			sycl::free(d_work, q);
			if(d_wall_site_indices) sycl::free(d_wall_site_indices, q);
		}

		// flat interface: phi=+1 for y<y_level, phi=-1 for y>=y_level (or
		// the straight tilted wall, see misistema.h)
		void set_flat_interface(int y_level)
		{
			if(tilt==0){
				int split=y_level*Lx;
				dpl::fill(policy, d_phi, d_phi+split, 1.0f);
				dpl::fill(policy, d_phi+split, d_phi+L2, -1.0f);
				return;
			}
			dpl::transform(policy, site(0), site(L2), d_phi, tilted_wall(Lx, y_level-std::min(tilt,0), tilt));
		};
		void set_flat_interface(){ set_flat_interface(y_start); }

		void set_geometry(int _y_start, int _escape_row, bool _escape_full)
		{
			y_start=_y_start; escape_row=_escape_row; escape_full=_escape_full;
		}
		void set_tilt(int m)
		{
			if(m%2!=0){
				std::cout << "ERROR: the tilt must be even (checkerboard colors across the x boundary)" << std::endl;
				exit(1);
			}
			tilt=m;
			std::ofstream log("logfile.dat",std::ios::app);
			log << "tilt= " << tilt << std::endl;
		}
		int get_tilt(){ return tilt; }
		int get_y_start(){ return y_start; }
		int get_escape_row(){ return escape_row; }
		bool get_escape_full(){ return escape_full; }

		void reset_uniform(){ dpl::fill(policy, d_phi, d_phi+L2, -1.0f); };

		// keep a copy of phi / go back to it (warm starts of the bisection)
		void save_state(){ q.memcpy(d_saved, d_phi, sizeof(float)*L2).wait(); }
		void restore_state(){ q.memcpy(d_phi, d_saved, sizeof(float)*L2).wait(); }
		void swap_state(){ dpl::swap_ranges(policy, d_phi, d_phi+L2, d_saved); }

		// max_i |phi_i - other_i|
		float max_diff(float *other)
		{
			return dpl::transform_reduce(policy, d_phi, d_phi+L2, other, 0.0f,
				dpl::maximum<float>(), abs_diff());
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
			unsigned long long w;
			q.memcpy(&w, d_work, sizeof(w)).wait();
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
					vmcop op(phi_ptr,Lx,Ly,tilt,external_field,color_offset,disorder_seed);
					q.parallel_for(sycl::range<1>(half_L2), [=](sycl::id<1> j){ op(int(j[0])); });
				}
				total_time++;
			}
			q.wait();
		}

		// the same sweeps, by active tiles. The first sweep of every call
		// (both colors) updates all tiles, since phi or h may have been
		// changed in between.
		void dynamics_tiles(int num_sweeps)
		{
			int Tx=Lx/TILE, Ty=Ly/TILE, ntiles=Tx*Ty;
			// locals, so that the kernels capture values, not this
			unsigned char *active=d_active, *changed=d_changed;
			unsigned long long *work=d_work;
			float *phi=phi_ptr, h=external_field, eps=tile_eps;
			int lx=Lx, ly=Ly, m=tilt;
			unsigned long seed=disorder_seed;
			const int group=TILE*TILE/2;
			q.memset(active,1,ntiles);
			for(int sweep=0;sweep<num_sweeps;sweep++)
			{
				for(int color_offset=0;color_offset<2;color_offset++){
					q.memset(changed,0,ntiles);
					q.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(ntiles)*group), sycl::range<1>(group)),
						[=](sycl::nd_item<1> item){
							vmc_tiles_kernel(item,phi,lx,ly,m,h,color_offset,seed,active,changed,work,eps);
						});
					if(sweep==0 && color_offset==0) q.memset(active,1,ntiles);
					else q.parallel_for(sycl::range<1>(ntiles), [=](sycl::id<1> t){
						spread_activity(int(t[0]),changed,active,Tx,Ty,m);
					});
				}
				total_time++;
			}
			q.wait();
		}

		// mean residual velocity <dphi/dt>, normalized by Lx (see misistema.h)
		float velocity_phi()
		{
			return dpl::transform_reduce(policy, site(0), site(L2), float(0.0), std::plus<float>(),
				velocityop(phi_ptr,Lx,Ly,tilt,external_field,disorder_seed))/float(Lx);
		}

		// largest residual max_i |dphi_i/dt|
		float max_residual()
		{
			return dpl::transform_reduce(policy, site(0), site(L2), 0.0f, dpl::maximum<float>(),
				abs_velocityop(velocityop(phi_ptr,Lx,Ly,tilt,external_field,disorder_seed)));
		}

		// relaxes to a metastable state (see misistema.h)
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
						q.memcpy(d_snap, d_phi, sizeof(float)*L2).wait();
						snap_iteration=iterations;
						moving=true;
					}
					else if((iterations-snap_iteration)*chop<CONFIRM_SWEEPS) moving=true;
					else if(max_diff(d_snap)>CONFIRM_DPHI){
						q.memcpy(d_snap, d_phi, sizeof(float)*L2).wait();
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

		float magnetization(){ return dpl::reduce(policy, d_phi, d_phi+L2, 0.0f); }

		// max / min of phi along row y
		float row_max(int y){ return dpl::reduce(policy, d_phi+Lx*y, d_phi+Lx*(y+1), float(-10.), dpl::maximum<float>()); }
		float row_min(int y){ return dpl::reduce(policy, d_phi+Lx*y, d_phi+Lx*(y+1), float(10.), dpl::minimum<float>()); }

		// has the interface escaped (see escape_row)?
		bool escaped(){
			if(!escape_full) return row_max(escape_row)>0;
			return row_min(escape_row)>0 || row_min((escape_row+Ly)/2)>0;
		}

		// rows spanned by the wall: (lowest row with some phi<0, highest
		// row with some phi>0)
		std::pair<int,int> wall_rows()
		{
			int bottom=dpl::transform_reduce(policy, site(0), site(L2), Ly, dpl::minimum<int>(),
				row_if_sign(phi_ptr,Lx,-1,Ly));
			int top=dpl::transform_reduce(policy, site(0), site(L2), -1, dpl::maximum<int>(),
				row_if_sign(phi_ptr,Lx,+1,-1));
			return std::make_pair(bottom,top);
		}

		float local_magnetization(int x, int y){
			float v;
			q.memcpy(&v, d_phi+(x+Lx*y), sizeof(float)).wait();
			return v;
		}

		std::tuple<int,int,int,int,int> detect_wall(int x1, int x2, int y1, int y2)
		{
			// counted first, so the buffer is only as large as the wall
			is_wall_smooth is_wall(phi_ptr,Lx,Ly,tilt,x1,x2,y1,y2);
			int num_wall_points = dpl::count_if(policy, site(0), site(L2), is_wall);
			if(d_wall_site_indices) sycl::free(d_wall_site_indices, q);
			d_wall_site_indices = sycl::malloc_device<int>(std::max(num_wall_points,1), q);
			num_wall_site_indices = num_wall_points;
			dpl::copy_if(policy, site(0), site(L2), d_wall_site_indices, is_wall);

			int area_positive = dpl::count_if(policy, d_phi, d_phi+L2, is_positive());
			int area_negative = L2 - area_positive;

			// copy_if keeps the order, so the indices come out sorted
			wall_first_index=-1; wall_last_index=-1;
			if(num_wall_points){
				q.memcpy(&wall_first_index, d_wall_site_indices, sizeof(int));
				q.memcpy(&wall_last_index, d_wall_site_indices+num_wall_points-1, sizeof(int));
				q.wait();
			}

			int last_y=int(wall_last_index/Lx);
			int first_y=int(wall_first_index/Lx);

			return std::tuple<int,int,int,int,int>(num_wall_points,area_positive,area_negative,first_y,last_y);
		}

		// saves the domain-wall coordinates restricted to x in (x1,x2), y in (y1,y2)
		int print_wall(std::ofstream &fout, int x1, int x2, int y1, int y2)
		{
			std::tuple<int,int,int,int,int> stats = detect_wall(x1, x2, y1, y2);
			int num_wall_points=std::get<0>(stats);
			int area_positive=std::get<1>(stats);
			int area_negative=std::get<2>(stats);

			std::cout << "# " << num_wall_points << " " << area_positive << " " << area_negative
				<< " " << wall_first_index << " " << wall_last_index << std::endl;
			fout << "# " << num_wall_points << " " << area_positive << " " << area_negative
				<< " " << wall_first_index << " " << wall_last_index << std::endl;

			std::vector<int> wall_points(num_wall_points);
			if(num_wall_points) q.memcpy(wall_points.data(), d_wall_site_indices, sizeof(int)*num_wall_points).wait();
			sycl::free(d_wall_site_indices, q);
			d_wall_site_indices=nullptr;
			num_wall_site_indices=0;

			for(int i=0;i<num_wall_points;i++){
				int x=wall_points[i]%Lx;
				int y=int(wall_points[i]/Lx);
				fout << x << " " << y << std::endl;
			}
			fout << std::endl << std::endl;

			return num_wall_points;
		}

		// whole field, copied to the host
		std::vector<float> host_phi()
		{
			std::vector<float> h_phi(L2);
			q.memcpy(h_phi.data(), d_phi, sizeof(float)*L2).wait();
			return h_phi;
		}

		// per-column and per-row magnetization profile of the whole field
		void print_magnetization_profile(std::ofstream &fout)
		{
			std::vector<float> h_phi=host_phi();
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

		// raw float32 dump of the whole field, row-major (index x+y*Lx)
		void print_field_binary(std::ofstream &fout)
		{
			std::vector<float> h_phi=host_phi();
			fout.write(reinterpret_cast<const char*>(h_phi.data()),sizeof(float)*L2);
		}

		// reads a field written by print_field_binary
		void read_field_binary(std::ifstream &fin)
		{
			std::vector<float> h_phi(L2);
			fin.read(reinterpret_cast<char*>(h_phi.data()),sizeof(float)*L2);
			q.memcpy(d_phi, h_phi.data(), sizeof(float)*L2).wait();
		}

		void print_config(std::ofstream &fout)
		{
			std::vector<float> h_phi=host_phi();
			for(int i=0;i<Ly;i++){
				for(int j=0;j<Lx-1;j++) fout << h_phi[j+i*Lx] << " ";
				fout << h_phi[Lx-1+i*Lx] << "\n";
			}
			fout << "\n" << std::endl;
		}

		void print_config_linear(std::ofstream &fout)
		{
			std::vector<float> h_phi=host_phi();
			for(int i=0;i<L2;i++) fout << h_phi[i] << std::endl;
			fout << "\n" << std::endl;
		}

		void read_config_linear(std::ifstream &fin)
		{
			std::vector<float> h_phi(L2);
			for(int i=0;i<L2;i++) fin >> h_phi[i];
			q.memcpy(d_phi, h_phi.data(), sizeof(float)*L2).wait();
		}
};
