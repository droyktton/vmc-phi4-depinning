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
#include <cassert>
#include <iostream>
#include <cmath>
#include <fstream>

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

#define MAXITERATIONS	1000000

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
	int L;
	float * phi;
	float external_field;
	unsigned long disorder_seed;

	velocityop(float *_phi, int _L, float _external_field, unsigned long _disorder_seed):
	phi(_phi),L(_L),external_field(_external_field),disorder_seed(_disorder_seed){};

	__device__
	float operator()(int i)
	{
		int x=i%L;
		int y=int(i/L);

		int y_plus1=(y+1)%L, y_minus1=(y-1+L)%L;
		int x_plus1=(x+1)%L, x_minus1=(x-1+L)%L;

		int up=x+L*y_minus1, down=x+L*y_plus1;
		int right=x_plus1+L*y, left=x_minus1+L*y;
		int center=x+L*y;

		float phi_up=phi[up], phi_down=phi[down];
		float phi_right=phi[right], phi_left=phi[left];
		float phi_center=phi[center];

		// anti-periodic BC in y: a single domain wall spans the sample
		if(y==0) phi_down*=-1;
		if(y==L-1) phi_up*=-1;

		float laplacian = phi_up+phi_down+phi_right+phi_left-4.0*phi_center;
		float r = site_disorder(center, disorder_seed);
		float phi4_force = EPSILON0*(R0*(1.0+r)*phi_center - phi_center*phi_center*phi_center);

		return CEL*laplacian + phi4_force + external_field;
	}
};

// elementary VMC move: sets phi_ij to the root of the steady-state cubic
// closest to its current value. Sites are dispatched in a checkerboard
// (2-coloring) pattern so that a whole color can be updated in parallel:
// each site's four neighbors always belong to the other color.
struct vmcop
{
	int L;
	float * phi;
	float external_field;
	int color_offset;
	unsigned long disorder_seed;

	vmcop(float *_phi, int _L, float _external_field, int _color_offset, unsigned long _disorder_seed)
	:phi(_phi),L(_L),external_field(_external_field),color_offset(_color_offset),disorder_seed(_disorder_seed)
	{};

	__device__
	void operator()(int j)
	{
		int i = (2*j) + (int((2*j)/L)+color_offset)%2;

		int x=i%L;
		int y=int(i/L);

		int y_plus1=(y+1)%L, y_minus1=(y-1+L)%L;
		int x_plus1=(x+1)%L, x_minus1=(x-1+L)%L;

		int up=x+L*y_minus1, down=x+L*y_plus1;
		int right=x_plus1+L*y, left=x_minus1+L*y;
		int center=x+L*y;

		float phi_up=phi[up], phi_down=phi[down];
		float phi_right=phi[right], phi_left=phi[left];
		float phi_center=phi[center];

		// anti-periodic BC in y: a single domain wall spans the sample
		if(y==0) phi_down*=-1;
		if(y==L-1) phi_up*=-1;

		float r = site_disorder(center, disorder_seed);
		float r_center = R0*(1.0+r);

		/* Vieta form: phi^3 + p*phi - q == 0, derived from
		   0 = c*Laplacian(phi) + eps0*[(1+r)*phi - phi^3] + h
		   -> p = 4*c/eps0 - r_center ; q = (c*sum_neighbors + h)/eps0 */
		float p = 4.0*CEL/EPSILON0 - r_center;
		float q = (CEL*(phi_up+phi_down+phi_right+phi_left) + external_field)/EPSILON0;

		float x1,x2,x3;
		int num_roots = vieta_solver(p, q, x1, x2, x3);

		if(num_roots==1) phi[center]=x1;
		else{
			// pick the root closest to the current value
			if(phi_center>x3) phi[center]=x1;
			else phi[center]=x2;
		}
	}
};

__host__ __device__ int index_center(int x,int y,int L){ return x+L*y; }
__host__ __device__ int index_up(int x,int y,int L){ return x+L*((y-1+L)%L); }
__host__ __device__ int index_down(int x,int y,int L){ return x+L*((y+1)%L); }
__host__ __device__ int index_right(int x,int y,int L){ return (x+1)%L+L*y; }
__host__ __device__ int index_left(int x,int y,int L){ return (x-1+L)%L+L*y; }

// detects domain-wall sites, restricted to the [x1,x2]x[y1,y2] window
// (to avoid picking up the periodic-image seam). Each site is first
// replaced by its 5-point (self+4 neighbors) smoothed value, and a wall
// site is one where the smoothed value changes sign between the row
// above and the row below -- this smoothing is what lets a single,
// well-defined interface height be extracted even where the raw
// configuration has overhangs/pinch-off loops (see paper, Fig. 1).
struct is_wall_smooth
{
	int L;
	float *phi;
	int x1, x2, y1, y2;
	is_wall_smooth(float *_phi, int _L, int _x1, int _x2, int _y1, int _y2):
	phi(_phi),L(_L),x1(_x1),x2(_x2),y1(_y1),y2(_y2){};

	__device__
	float smoothed(int x, int y)
	{
		return phi[index_center(x,y,L)]+phi[index_up(x,y,L)]+phi[index_down(x,y,L)]
			+phi[index_right(x,y,L)]+phi[index_left(x,y,L)];
	}

	__device__
	bool operator()(int i)
	{
		int x=i%L;
		int y=int(i/L);

		float smoothed_up=smoothed(x,(y-1+L)%L);
		float smoothed_down=smoothed(x,(y+1)%L);

		bool sign_change = (smoothed_up*smoothed_down<0);
		return sign_change && x>x1 && x<x2 && y>y1 && y<y2;
	}
};

struct is_positive
{
	__host__ __device__
	bool operator()(float x){ return x > 0; }
};

class System{
	private:
		thrust::device_ptr<float> d_phi;
		thrust::device_ptr<int> d_wall_site_indices; // scratch buffer for wall-site indices
		int wall_first_index;
		int wall_last_index;

		int L;
		int L2;

		float external_field;
		int total_time;

	public:
		unsigned long disorder_seed;
		float *phi_ptr;

		System(int _L, unsigned long _disorder_seed)
		{
			if(_L%2!=0){
				std::cout << "ERROR: L must be even for checkerboard VMC" << std::endl;
				exit(1);
			}

			L=_L;
			L2=L*L;

			d_phi = thrust::device_malloc<float>(L2);
			phi_ptr = thrust::raw_pointer_cast(&d_phi[0]);
			d_wall_site_indices = thrust::device_malloc<int>(L*10);

			external_field=0.0;
			total_time=0;
			disorder_seed=_disorder_seed;

			reset_uniform();

			std::ofstream log("logfile.dat");
			log << "L= " << L << std::endl;
			log << "disorder_seed= " << disorder_seed << std::endl;
			log << "AMPDIS (Delta)= " << AMPDIS << std::endl;
			log << "CEL (c)= " << CEL << std::endl;
			log << "EPSILON0= " << EPSILON0 << std::endl;
			log << "R0= " << R0 << std::endl;
			log << "TOLVEL (epsilon)= " << TOLVEL << std::endl;
		}

		~System(){
			thrust::device_free(d_phi);
			thrust::device_free(d_wall_site_indices);
		}

		// flat interface: phi=+1 for y<y_level, phi=-1 for y>=y_level
		void set_flat_interface(int y_level)
		{
			int split=y_level*L;
			thrust::fill(d_phi, d_phi+split, 1);
			thrust::fill(d_phi+split, d_phi+L2, -1);
		};
		void set_flat_interface(){ set_flat_interface(int(L*0.2)); }

		void reset_uniform(){ thrust::fill(d_phi, d_phi+L2, -1.0f); };

		void set_external_field(float h){ external_field=h; }
		float get_external_field(){ return external_field; }
		int get_L(){ return L; }
		int get_total_time(){ return total_time; }

		// checkerboard VMC sweeps: num_sweeps full sweeps of the lattice
		void dynamics(int num_sweeps)
		{
			int half_L2=L2/2;
			for(int sweep=0;sweep<num_sweeps;sweep++)
			{
				for(int color_offset=0;color_offset<2;color_offset++){
					thrust::for_each(
						thrust::make_counting_iterator(0),
						thrust::make_counting_iterator(half_L2),
						vmcop(phi_ptr,L,external_field,color_offset,disorder_seed)
					);
				}
				total_time++;
			}
		}

		// mean residual velocity <dphi/dt>, i.e. the paper's convergence
		// diagnostic (note: normalized by L, not L2, matching the scale
		// at which TOLVEL was tuned)
		float velocity_phi()
		{
			return thrust::transform_reduce(
				thrust::make_counting_iterator(0),
				thrust::make_counting_iterator(L2),
				velocityop(phi_ptr,L,external_field,disorder_seed),
				float(0.0), thrust::plus<float>()
			)/float(L);
		}

		// runs elementary chunks of "chop" VMC sweeps until the mean DW
		// velocity drops below TOLVEL (i.e. a metastable state has been
		// reached)
		void find_next_metastable(int chop)
		{
			float vel;
			int iterations=0;
			do{
				dynamics(chop);
				vel=velocity_phi();
				iterations++;
			}while(fabs(vel)>TOLVEL && iterations<MAXITERATIONS);

			if(iterations>10) std::cout << "\nblocked after " << iterations << " VMC iterations" << std::endl;
		}

		float magnetization(){ return thrust::reduce(d_phi,d_phi+L2); }

		// max(phi) along row y=relative_y*L: >0 signals the interface has
		// swept past that row (used to bracket the depinning field)
		float magnetization_lane(float relative_y)
		{
			int y=relative_y*L;
			return thrust::reduce(d_phi+L*y,d_phi+L*(y+1),float(-10.),thrust::maximum<float>());
		}

		float local_magnetization(int x, int y){ return d_phi[(x+L*y)]; }

		thrust::tuple<int,int,int,int,int> detect_wall(int x1, int x2, int y1, int y2)
		{
			int num_wall_points = int(
				thrust::copy_if(
					thrust::make_counting_iterator(0),
					thrust::make_counting_iterator(L2),
					d_wall_site_indices, is_wall_smooth(phi_ptr,L,x1,x2,y1,y2)
				) - d_wall_site_indices
			);

			int area_positive = thrust::count_if(thrust::device, phi_ptr, phi_ptr+L2, is_positive());
			int area_negative = L2 - area_positive;

			thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int> > result =
				thrust::minmax_element(d_wall_site_indices, d_wall_site_indices+num_wall_points);

			wall_last_index=(result.second)[0];
			wall_first_index=(result.first)[0];

			int last_y=int(wall_last_index/L);
			int first_y=int(wall_first_index/L);

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

			thrust::host_vector<int> wall_points(num_wall_points);
			thrust::copy(d_wall_site_indices,d_wall_site_indices+num_wall_points,wall_points.begin());

			for(int i=0;i<num_wall_points;i++){
				int x=wall_points[i]%L;
				int y=int(wall_points[i]/L);
				fout << x << " " << y << std::endl;
			}
			fout << std::endl << std::endl;

			return num_wall_points;
		}

		// per-row and per-column magnetization profile of the whole field
		void print_magnetization_profile(std::ofstream &fout)
		{
			thrust::device_vector<int> x_index(L2);
			thrust::device_vector<int> y_index(L2);
			thrust::device_vector<float> mag_sorted(d_phi,d_phi+L2);
			thrust::device_vector<float> mag_by_x(L);
			thrust::device_vector<float> mag_by_y(L);

			using namespace thrust::placeholders;
			thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(L2),x_index.begin(),_1%L);
			thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(L2),y_index.begin(),_1/L);

			thrust::reduce_by_key(y_index.begin(),y_index.end(),mag_sorted.begin(),thrust::make_discard_iterator(),mag_by_y.begin());

			thrust::sort_by_key(x_index.begin(),x_index.end(),mag_sorted.begin());
			thrust::reduce_by_key(x_index.begin(),x_index.end(),mag_sorted.begin(),thrust::make_discard_iterator(),mag_by_x.begin());

			for(int i=0;i<L;i++) fout << mag_by_x[i] << " " << mag_by_y[i] << std::endl;
			fout << "\n" << std::endl;
		}

		void print_config(std::ofstream &fout)
		{
			for(int i=0;i<L;i++){
				for(int j=0;j<L-1;j++) fout << d_phi[j+i*L] << " ";
				fout << d_phi[L-1+i*L] << "\n";
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
