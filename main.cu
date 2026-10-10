// Finds the depinning field h_d and the critical (depinning) interface
// configuration of the disordered phi^4 model, for one disorder
// realization ("sample"), using the variant Monte Carlo (VMC) solver
// implemented in misistema.h.
//
// Reference: arXiv:2306.13415 ("Depinning without the elastic
// approximation: pinch-off, overhangs and structure factor").
//
// Method: bisection on the external field h. For each trial h, the
// interface is relaxed to a metastable state; escape is detected as the
// lane at ESCAPE_LANE*Ly flipping sign, at which point that trial's
// relaxation is stopped early. Bisection stops at |h_high-h_low|<tol,
// giving h_d = h_low (the largest field found pinned).
//
// Warm starts: the VMC map is monotone (no-passing), so the metastable
// state M(h) reached from a flat interface is also reached from any
// state X with flat <= X <= M(h). The pinned state M(a) at a field a<h
// is such a state, so each trial starts from the last pinned state
// instead of from the flat interface: the wall only has to move from
// M(a) to M(h), not all the way from y=0.2Ly. After an escape, the last
// pinned state is restored. (flat <= M(a) only holds up to the bulk
// sites, which relax within a few sweeps, so this is checked: at the end,
// --verify-cold relaxes again from the flat interface at h_d and
// compares; see docs/analysis_math.md, section 2.3.)
//
// Geometry: Lx (along the wall, periodic) by Ly (along the motion,
// anti-periodic). The wall starts flat at y=0.2Ly and escapes at
// 0.95Ly, so it can explore 0.75Ly. Since the wall's vertical extent
// grows as Lx^zeta, finite-size scaling needs Ly = k Lx^zeta at fixed k
// (--ly-k; zeta = LY_ZETA = 1.25, qEW). A square box (the default) has
// k = Lx^-0.25, which shrinks with Lx.
//
// Around h_d, scan_magnetization_jumps() additionally records the
// metastable-state sequence (magnetization jumps / "avalanches") on a
// finer field grid. This scan probes 100 fields all within 5% of h_d,
// i.e. deliberately in the critical-slowing-down region, so it is
// comparatively expensive; it is opt-in (--with-magjumps) since h_d
// and the critical configuration are already fully determined and
// saved by find_depinning_field() on its own.
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <chrono>
#include "misistema.h"

// a trial field is "depinned" once the interface sweeps past this row
// (relative height); relaxation stops early as soon as it does
#define ESCAPE_LANE 0.95

// relaxation is checked every CHOP sweeps
#define CHOP 1000

// Ly = k Lx^LY_ZETA with --ly-k k (rounded to a multiple of 32, the tile)
#define LY_ZETA 1.25

// relaxes the current state at field h; true if the interface escaped,
// or (with the default, local stopping criterion) if it was still moving
// after MAXSWEEPS sweeps (see misistema.h)
bool depins(System *sys, float h, const char *stage)
{
	std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now();
	double work0=sys->tile_work();
	sys->set_external_field(h);
	int sweeps=sys->find_next_metastable(CHOP,ESCAPE_LANE);
	bool escaped=sys->magnetization_lane(ESCAPE_LANE)>0;
	bool capped=!escaped && !sys->converged();
	if(capped && sys->get_tolres()>0) escaped=true;
	double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
	std::cout << stage << " h=" << h << (escaped ? " depinned" : " pinned") << (capped ? " (still moving at the cap)" : "")
		<< " sweeps=" << sweeps << " time=" << seconds << "s";
	if(sys->get_use_tiles()) std::cout << " tile_work=" << sys->tile_work()-work0 << " sweeps";
	std::cout << std::endl;
	return escaped;
}

// bisects on h to find the depinning field h_d for the sample seeded by
// sys->disorder_seed. [h_low,h_high] needn't bracket h_d: it is widened
// (h_low/1.5, h_high*1.5) until it does. Saves the critical
// configuration (critica_h..._seed....dat) and its magnetization profile
// (criticamag_h..._seed....dat), and, if dump_field, the full field as
// raw float32 (criticafield_h..._seed....bin). Returns h_d.
float find_depinning_field(float h_low, float h_high, float tol, System *sys, bool dump_field, bool verify_cold, bool no_warm)
{
	float a=h_low, b=h_high, mid;

	// lower end, from a flat interface
	sys->set_flat_interface();
	while(depins(sys,a,"low")){
		if(a<=0){
			std::cerr << "ERROR: the interface depins at h=" << a << std::endl;
			exit(1);
		}
		b=a;
		a/=1.5;
		sys->set_flat_interface();
	}
	sys->save_state();

	// upper end, warm-started from the pinned state at a
	if(no_warm) sys->set_flat_interface();
	while(!depins(sys,b,"high")){
		a=b;
		sys->save_state();
		b*=1.5;
		if(no_warm) sys->set_flat_interface();
	}
	sys->restore_state();

	int iteration=0;
	while(b-a>tol && iteration<100)
	{
		mid=(a+b)*0.5;
		if(no_warm) sys->set_flat_interface();
		if(depins(sys,mid,"bisect")){
			b=mid;
			sys->restore_state();
		}
		else{
			a=mid;
			sys->save_state();
		}
		std::cout << "[" << a << "," << b << "]" << std::endl;
		iteration++;
	}
	// phi (and the saved copy) now hold the pinned state at a

	if(verify_cold){
		sys->set_flat_interface();
		bool escaped=depins(sys,a,"cold");
		std::cout << "cold-start check at h=" << a << ": " << (escaped ? "DEPINNED" : "pinned")
			<< ", max|phi_warm-phi_cold| = " << sys->max_diff_saved() << std::endl;
		sys->restore_state();
	}

	char filename[100];
	int Lx=sys->get_L(), Ly=sys->get_Ly();

	sprintf(filename,"critica_h%f_seed%lu.dat",a,sys->disorder_seed);
	std::ofstream critical_wall_out(filename);
	sys->print_wall(critical_wall_out,0,Lx,int(Ly*0.01),int(Ly*0.99));

	sprintf(filename,"criticamag_h%f_seed%lu.dat",a,sys->disorder_seed);
	std::ofstream critical_mag_out(filename);
	sys->print_magnetization_profile(critical_mag_out);

	if(dump_field){
		sprintf(filename,"criticafield_h%f_seed%lu.bin",a,sys->disorder_seed);
		std::ofstream critical_field_out(filename,std::ios::binary);
		sys->print_field_binary(critical_field_out);
	}

	return a;
}

// scans h on a fine linear grid from h_low to h_high (typically just
// below h_d to h_d), recording the magnetization jump at each step --
// this is the avalanche/metastable-state-sequence statistics discussed
// in the paper.
void scan_magnetization_jumps(float h_low, float h_high, System *sys)
{
	int Lx=sys->get_L(), Ly=sys->get_Ly();
	double N=double(Lx)*Ly;
	char filename[100];

	sprintf(filename,"jumps_seed%lu.dat",sys->disorder_seed);
	std::ofstream jumps_out(filename);
	sprintf(filename,"metas_seed%lu.dat",sys->disorder_seed);
	std::ofstream metastable_wall_out(filename);
	sprintf(filename,"metasmag_seed%lu.dat",sys->disorder_seed);
	std::ofstream metastable_mag_out(filename);

	sys->set_flat_interface();
	sys->set_external_field(h_low);
	sys->find_next_metastable(100);

	float previous_magnetization=sys->magnetization()/N;
	int num_steps=100;
	float h=h_low;
	for(int step=1;step<=num_steps;step++){
		h+=(h_high-h_low)/float(num_steps);
		sys->set_external_field(h);
		sys->find_next_metastable(100);
		float magnetization=sys->magnetization()/N;
		jumps_out << step << " " << h << " " << h_high << " " << magnetization-previous_magnetization << std::endl;
		previous_magnetization=magnetization;
		sys->print_wall(metastable_wall_out,0,Lx,int(Ly*0.01),int(Ly*0.99));
		sys->print_magnetization_profile(metastable_mag_out);
	}
}

int main(int argc, char **argv)
{
	int L; float h_low,h_high,tol; unsigned long disorder_seed;
	bool with_magjumps=false, dump_field=false, verify_cold=false, bad_flag=false;
	float tolres=-1, tile_eps=-1, ly_k=-1;
	int Ly=-1;
	std::string relax_from;
	bool no_warm=false, no_tiles=false;
	for(int i=6;i<argc;i++){
		std::string flag(argv[i]);
		if(flag=="--with-magjumps") with_magjumps=true;
		else if(flag=="--dump-field") dump_field=true;
		else if(flag=="--verify-cold") verify_cold=true;
		else if(flag=="--no-warm") no_warm=true;
		else if(flag=="--no-tiles") no_tiles=true;
		else if(flag=="--tile-eps" && i+1<argc) tile_eps=atof(argv[++i]);
		else if(flag=="--relax-from" && i+1<argc) relax_from=argv[++i];
		else if(flag=="--tolres" && i+1<argc) tolres=atof(argv[++i]);
		else if(flag=="--ly" && i+1<argc) Ly=atoi(argv[++i]);
		else if(flag=="--ly-k" && i+1<argc) ly_k=atof(argv[++i]);
		else bad_flag=true;
	}
	if(argc>=6 && !bad_flag){
		L=atoi(argv[1]);
		h_low=atof(argv[2]);
		h_high=atof(argv[3]);
		tol=atof(argv[4]);
		disorder_seed=strtoul(argv[5],NULL,10);
	}
	else{
		std::cout << "usage: " << argv[0] << " L h_low h_high tol seed [--with-magjumps] [--dump-field] [--verify-cold]" << std::endl;
		std::cout << "  L               lattice size Lx along the wall (must be even)" << std::endl;
		std::cout << "  h_low           lower bracket for the depinning field (lowered if the interface depins there)" << std::endl;
		std::cout << "  h_high          upper bracket for the depinning field (raised if the interface stays pinned there)" << std::endl;
		std::cout << "  tol             bisection tolerance on h_d (paper uses 1e-4)" << std::endl;
		std::cout << "  seed            disorder-realization seed" << std::endl;
		std::cout << "  --with-magjumps also run the (comparatively expensive) magnetization-jump scan near h_d" << std::endl;
		std::cout << "  --dump-field    also save the critical field phi as raw float32 (Lx*Ly*4 bytes)" << std::endl;
		std::cout << "  --verify-cold   relax again from a flat interface at h_d and compare with the warm-started state" << std::endl;
		std::cout << "  --no-warm       start every trial from a flat interface (no warm starts)" << std::endl;
		std::cout << "  --no-tiles      sweep the whole lattice instead of the active tiles only" << std::endl;
		std::cout << "  --tile-eps x    a change |dphi|<=x doesn't keep a tile active (default " << TILE_EPS << "; 0: exact)" << std::endl;
		std::cout << "  --relax-from f  instead: relax the field in f (as saved by --dump-field) at h=h_low, and report" << std::endl;
		std::cout << "                  how far it moves (checks that a saved configuration is metastable)" << std::endl;
		std::cout << "  --ly N          lattice size Ly along the motion (default: Lx)" << std::endl;
		std::cout << "  --ly-k k        Ly = k*Lx^" << LY_ZETA << ", rounded to a multiple of 32 (keeps the box's" << std::endl;
		std::cout << "                  shape scaled with the wall's roughness across Lx)" << std::endl;
		std::cout << "  --tolres x      convergence cutoff on max|dphi/dt| (default " << TOLRES << "; 0: legacy mean-velocity criterion)" << std::endl;
		std::cout << "example: " << argv[0] << " 256 0.0 0.1 0.0001 1234" << std::endl;
		return 1;
	}

	if(ly_k>0) Ly=32*std::max(1,int(lround(ly_k*pow(double(L),LY_ZETA)/32)));
	if(Ly<=0) Ly=L;
	std::cout << "Lx=" << L << " Ly=" << Ly << std::endl;
	System sys(L,Ly,disorder_seed);
	if(tolres>=0) sys.set_tolres(tolres);
	if(no_tiles) sys.set_use_tiles(false);
	if(tile_eps>=0) sys.set_tile_eps(tile_eps);

	if(!relax_from.empty()){
		std::ifstream fin(relax_from.c_str(),std::ios::binary);
		if(!fin){ std::cerr << "ERROR: cannot read " << relax_from << std::endl; return 1; }
		sys.read_field_binary(fin);
		sys.save_state();
		bool escaped=depins(&sys,h_low,"relax");
		float dphi=sys.max_diff_saved();
		sys.swap_state();
		float positive_before=sys.magnetization();
		sys.swap_state();
		std::cout << "relaxed " << relax_from << " at h=" << h_low << ": " << (escaped ? "DEPINNED" : "pinned")
			<< ", max|dphi| = " << dphi << ", d(sum phi)/L = " << (sys.magnetization()-positive_before)/L << std::endl;
		return 0;
	}

	float hd = find_depinning_field(h_low, h_high, tol, &sys, dump_field, verify_cold, no_warm);
	std::cout << "depinning field h_d = " << hd << std::endl;

	if(with_magjumps) scan_magnetization_jumps(hd*0.95, hd, &sys);

	return 0;
}
