// Finds the depinning field h_d and the critical (depinning) interface
// configuration of the disordered phi^4 model, for one disorder
// realization ("sample"), using the variant Monte Carlo (VMC) solver
// implemented in misistema.h.
//
// Reference: arXiv:2306.13415 ("Depinning without the elastic
// approximation: pinch-off, overhangs and structure factor").
//
// Method: bisection on the external field h. For each trial h, the
// interface is relaxed to a metastable state; escape is detected at
// the row escape_row (below), at which point that trial's relaxation is
// stopped early. Bisection stops at |h_high-h_low|<tol,
// giving h_d = h_low (the largest field found pinned).
//
// Warm starts: the VMC map is monotone (no-passing), so the metastable
// state M(h) reached from a flat interface is also reached from any
// state X with flat <= X <= M(h). The pinned state M(a) at a field a<h
// is such a state, so each trial starts from the last pinned state
// instead of from the flat interface: the wall only has to move from
// M(a) to M(h), not all the way from its starting row. After an escape, the last
// pinned state is restored. (flat <= M(a) only holds up to the bulk
// sites, which relax within a few sweeps, so this is checked: at the end,
// --verify-cold relaxes again from the flat interface at h_d and
// compares; see docs/analysis_math.md, section 2.3.)
//
// Geometry: Lx (along the wall, periodic) by Ly (along the motion,
// anti-periodic). By default the wall starts flat at y=0.2Ly and has
// escaped once some site of row 0.95Ly flips, so it explores 0.75Ly;
// Ly is --ly, or k Lx^zeta with --ly-k (zeta = LY_ZETA = 1.25, qEW).
// That censors wide critical walls: if the strongest pinning lies just
// below the escape row, a protrusion reaches it and the trial counts as
// depinned. With --path P --headroom H instead, the wall starts at
// y0 (--y0, default 32), escapes once *all* of row y0+P has flipped
// (or, if a pinned bubble of the old phase keeps some of its sites from
// flipping, all of the row halfway up the headroom), and Ly = y0+P+H: a pinned wall may stick out up to H above the escape
// row. P sets the finite-size regime (the aspect ratio P/w, w the
// wall's width; Kolton et al., arXiv:1308.4329), H must exceed the
// widest critical wall. The critical wall's rows are reported, with a
// WARNING if it comes within WALL_MARGIN rows of the escape row
// (legacy criterion) or of Ly (--path).
//
// Tilt: with --tilt m (even, needs --path) the x boundary is helical,
// phi(x+Lx,y) = phi(x,y-m), so the wall has a mean slope m/Lx. It starts
// as the straight wall from row y0 (its lowest row) rising by |m| over
// Lx, escapes once all of row y0+P has flipped, and Ly = y0+P+|m|+H.
// h_d versus m/Lx measures the tilt dependence of the threshold (a
// nonzero curvature is the qKPZ coupling at depinning).
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

// --y0 default, and how close (in rows) the critical wall may come to
// the escape row (legacy) or to Ly (--path) before a WARNING
#define Y0_DEFAULT 32
#define WALL_MARGIN 32

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
	int sweeps=sys->find_next_metastable(CHOP,true);
	bool escaped=sys->escaped();
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

	// how close the critical wall gets to where it would be cut off
	thrust::pair<int,int> rows=sys->wall_rows();
	int limit = sys->get_escape_full() ? Ly-1 : sys->get_escape_row();
	std::cout << "wall rows: bottom=" << rows.first << " top=" << rows.second
		<< " extent=" << rows.second-rows.first << " margin=" << limit-rows.second
		<< " (to " << (sys->get_escape_full() ? "Ly" : "the escape row") << ")" << std::endl;
	if(limit-rows.second<WALL_MARGIN)
		std::cout << "WARNING: critical wall within " << WALL_MARGIN << " rows of "
			<< (sys->get_escape_full() ? "Ly" : "the escape row") << ": possibly censored" << std::endl;

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
	int Ly=-1, path=-1, headroom=-1, y0=Y0_DEFAULT, tilt=0;
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
		else if(flag=="--path" && i+1<argc) path=atoi(argv[++i]);
		else if(flag=="--headroom" && i+1<argc) headroom=atoi(argv[++i]);
		else if(flag=="--y0" && i+1<argc) y0=atoi(argv[++i]);
		else if(flag=="--tilt" && i+1<argc) tilt=atoi(argv[++i]);
		else bad_flag=true;
	}
	// --path and --headroom go together, and replace --ly/--ly-k
	if((path>0)!=(headroom>0) || (path>0 && (Ly>0 || ly_k>0))) bad_flag=true;
	if(tilt!=0 && path<=0) bad_flag=true;
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
		std::cout << "                  (both: start at 0.2Ly, depinned once some site of row 0.95Ly flips)" << std::endl;
		std::cout << "  --path P --headroom H [--y0 Y]  instead: start at row Y (default " << Y0_DEFAULT << "), depinned once" << std::endl;
		std::cout << "                  all of row Y+P has flipped, Ly = Y+P+H rounded up to a multiple of 32" << std::endl;
		std::cout << "  --tilt m        (with --path) helical BC phi(x+L,y)=phi(x,y-m): mean wall slope m/L (m even);" << std::endl;
		std::cout << "                  Ly = Y+P+|m|+H" << std::endl;
		std::cout << "  --tolres x      convergence cutoff on max|dphi/dt| (default " << TOLRES << "; 0: legacy mean-velocity criterion)" << std::endl;
		std::cout << "example: " << argv[0] << " 256 0.0 0.1 0.0001 1234" << std::endl;
		return 1;
	}

	if(ly_k>0) Ly=32*std::max(1,int(lround(ly_k*pow(double(L),LY_ZETA)/32)));
	if(path>0) Ly=32*int((long(y0)+path+std::abs(tilt)+headroom+31)/32);
	if(Ly<=0) Ly=L;
	if(double(L)*Ly>=2147483648.0){
		std::cerr << "ERROR: Lx*Ly = " << double(L)*Ly << " overflows the 32-bit site index" << std::endl;
		return 1;
	}
	System sys(L,Ly,disorder_seed);
	if(path>0) sys.set_geometry(y0,y0+path,true);
	if(tilt!=0) sys.set_tilt(tilt);
	std::cout << "Lx=" << L << " Ly=" << Ly << " start=" << sys.get_y_start() << " escape_row=" << sys.get_escape_row()
		<< " criterion=" << (sys.get_escape_full() ? "whole row" : "any site");
	if(tilt!=0) std::cout << " tilt=" << tilt << " (slope " << double(tilt)/L << ")";
	if(AX!=1 || AY!=1) std::cout << " disorder blocks " << AX << "x" << AY;
	std::cout << std::endl;
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
