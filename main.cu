// Finds the depinning field h_d and the critical (depinning) interface
// configuration of the disordered phi^4 model, for one disorder
// realization ("sample"), using the variant Monte Carlo (VMC) solver
// implemented in misistema.h.
//
// Reference: arXiv:2306.13415 ("Depinning without the elastic
// approximation: pinch-off, overhangs and structure factor").
//
// Method: bisection on the external field h. For each trial h, the
// interface is relaxed to a metastable state starting from a flat
// initial condition (Middleton's theorems guarantee that whether the
// interface eventually escapes the sample or stays pinned depends on h
// alone, not on the relaxation path, so restarting from a flat interface
// at every trial h is valid). Escape is detected as the top lane of the
// lattice flipping sign, at which point that trial's relaxation is
// stopped early. Bisection stops at |h_high-h_low|<tol, giving h_d.
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
#include "misistema.h"

// a trial field is "depinned" once the interface sweeps past this row
// (relative height); relaxation stops early as soon as it does
#define ESCAPE_LANE 0.95

// bisects on h in [h_low,h_high] (tolerance tol) to find the depinning
// field h_d for the sample seeded by sys->disorder_seed. Saves the
// critical configuration (critica_h..._seed....dat) and its
// magnetization profile (criticamag_h..._seed....dat), and, if
// dump_field, the full field as raw float32 (criticafield_h..._seed....bin).
// Returns h_d.
float find_depinning_field(float h_low, float h_high, float tol, System *sys, bool dump_field)
{
	float a=h_low, b=h_high, mid;
	int iteration=0;
	while(b-a>tol && iteration<10000)
	{
		mid=(a+b)*0.5;
		sys->set_external_field(mid);
		sys->set_flat_interface();
		sys->find_next_metastable(100,ESCAPE_LANE);
		if(sys->magnetization_lane(ESCAPE_LANE)<0) a=mid;
		else b=mid;
		std::cout << "[" << a << "," << b << "]" << std::endl;
		iteration++;
	}

	// step back below a until the interface is pinned again, so we save
	// a genuine critical (depinning) configuration
	do{
		a=a-tol;
		sys->set_external_field(a);
		sys->set_flat_interface();
		sys->find_next_metastable(100,ESCAPE_LANE);
	}while(sys->magnetization_lane(ESCAPE_LANE)>0);

	char filename[100];
	int L=sys->get_L();

	sprintf(filename,"critica_h%f_seed%lu.dat",a,sys->disorder_seed);
	std::ofstream critical_wall_out(filename);
	sys->print_wall(critical_wall_out,0,L,int(L*0.01),int(L*0.99));

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
	int L=sys->get_L();
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

	float previous_magnetization=sys->magnetization()/(L*L);
	int num_steps=100;
	float h=h_low;
	for(int step=1;step<=num_steps;step++){
		h+=(h_high-h_low)/float(num_steps);
		sys->set_external_field(h);
		sys->find_next_metastable(100);
		float magnetization=sys->magnetization()/(L*L);
		jumps_out << step << " " << h << " " << h_high << " " << magnetization-previous_magnetization << std::endl;
		previous_magnetization=magnetization;
		sys->print_wall(metastable_wall_out,0,L,int(L*0.01),int(L*0.99));
		sys->print_magnetization_profile(metastable_mag_out);
	}
}

int main(int argc, char **argv)
{
	int L; float h_low,h_high,tol; unsigned long disorder_seed;
	bool with_magjumps=false, dump_field=false, bad_flag=false;
	for(int i=6;i<argc;i++){
		std::string flag(argv[i]);
		if(flag=="--with-magjumps") with_magjumps=true;
		else if(flag=="--dump-field") dump_field=true;
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
		std::cout << "usage: " << argv[0] << " L h_low h_high tol seed [--with-magjumps] [--dump-field]" << std::endl;
		std::cout << "  L               lattice size (must be even)" << std::endl;
		std::cout << "  h_low           lower bracket for the depinning field" << std::endl;
		std::cout << "  h_high          upper bracket for the depinning field (must be pinned at h_low, depinned at h_high)" << std::endl;
		std::cout << "  tol             bisection tolerance on h_d (paper uses 1e-4)" << std::endl;
		std::cout << "  seed            disorder-realization seed" << std::endl;
		std::cout << "  --with-magjumps also run the (comparatively expensive) magnetization-jump scan near h_d" << std::endl;
		std::cout << "  --dump-field    also save the critical field phi as raw float32 (L*L*4 bytes)" << std::endl;
		std::cout << "example: " << argv[0] << " 256 0.0 0.1 0.0001 1234" << std::endl;
		return 1;
	}

	System sys(L,disorder_seed);

	float hd = find_depinning_field(h_low, h_high, tol, &sys, dump_field);
	std::cout << "depinning field h_d = " << hd << std::endl;

	if(with_magjumps) scan_magnetization_jumps(hd*0.95, hd, &sys);

	return 0;
}
