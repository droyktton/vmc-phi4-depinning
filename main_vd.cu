// Quasi-static velocity-driven ensemble (v -> 0+) for the disordered
// phi^4 model, with the VMC solver of misistema.h.
//
// Instead of fixing the field and bisecting for h_d (main.cu), the wall
// is tied by a spring to a well center W that advances in steps dW: the
// field is h(y) = h_u - H0*(y - W), so a wall at mean height u feels the
// drive F = h_u + H0*(W - u), a spring of constant 2*H0 (the field acts
// on the wall with force 2h per unit length). At every step the wall
// relaxes to the next metastable state (no-passing holds, since h(y)
// grows with W everywhere), and F, u and the avalanche size are logged.
// For v -> 0+ this samples the critical force, F -> h_d as H0 -> 0, its
// fluctuations, its correlator in W and the avalanches, with a mass
// length L_m = sqrt(sigma/2H0) (sigma = 2 sqrt(2)/3 for c = eps0 = 1)
// above which the roughness saturates.
//
// h_u is a uniform part (about the expected h_d) so that the wall sits
// near W instead of h_d/H0 rows behind it; it doesn't change the physics.
// The wall advances through fresh disorder on a conveyor belt
// (System::shift_rows): once its mean height in the box is more than
// --recenter rows above Ly/2, the field and the disorder are shifted
// down by a multiple of 32 rows. W and u are reported in the lab frame.
//
// Output, vd_seed<seed>.dat, one line per step:
//   step W u F S sweeps converged ell height Sdown n_mv depth
// u = (area of the phi>0 phase)/Lx - 1/2 (the wall lies half a row above
// the last phi>0 row), S = change of that area (sites) over the step.
// The geometry of the step (System::avalanche_geometry, from the field
// before and after the relaxation): ell = columns with a flipped site
// (lateral extent of the avalanche), height = rows between its lowest and
// highest flipped site (+1), Sdown = sites that went from phi>0 to
// phi<=0 (0 if no-passing holds, so S is the number of flipped sites);
// and the wall after the step: n_mv = multivalued columns (2 or more
// branches: overhangs or bubbles), depth = largest distance between the
// first and last wall crossing of a column (the deepest overhang/bay).
// With --dump-every k, the field every k steps as raw float32
// (vdfield_step<step>_seed<seed>.bin, Lx*Ly, box frame).
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include "misistema.h"

// relaxation is checked every CHOP sweeps
#define CHOP 1000
// the wall may come no closer than this to the top of the box
#define WALL_MARGIN 32

int main(int argc, char **argv)
{
	if(argc<8){
		std::cout << "usage: " << argv[0] << " L Ly H0 h_u dW nsteps seed [--F0 f] [--recenter d] [--dump-every k] [--reflect-y]" << std::endl;
		std::cout << "  L          lattice size Lx along the wall (multiple of 32)" << std::endl;
		std::cout << "  Ly         box height along the motion (multiple of 32)" << std::endl;
		std::cout << "  H0         field gradient (spring 2*H0, mass length sqrt(0.47/H0))" << std::endl;
		std::cout << "  h_u        uniform field (about the expected h_d)" << std::endl;
		std::cout << "  dW         step of the well center, in rows" << std::endl;
		std::cout << "  nsteps     number of steps" << std::endl;
		std::cout << "  seed       disorder-realization seed" << std::endl;
		std::cout << "  --F0 f     initial drive (default h_u): the flat wall starts at Ly/2 with W = u + (f-h_u)/H0" << std::endl;
		std::cout << "  --recenter d   shift the box once the wall is more than d rows above Ly/2 (default 64)" << std::endl;
		std::cout << "  --dump-every k save the field every k steps" << std::endl;
		std::cout << "  --reflect-y    reflecting y boundary instead of anti-periodic (deep trenches left behind)" << std::endl;
		std::cout << "  --row-offset R the box starts at lab row R (default 0): the wall starts at lab row R+Ly/2" << std::endl;
		return 1;
	}
	int L=atoi(argv[1]), Ly=atoi(argv[2]);
	float H0=atof(argv[3]), h_u=atof(argv[4]);
	double dW=atof(argv[5]);
	long nsteps=atol(argv[6]);
	unsigned long seed=strtoul(argv[7],NULL,10);
	float F0=h_u;
	int recenter=64, dump_every=0;
	long row_offset0=0;
	bool reflect_y=false;
	for(int i=8;i<argc;i++){
		std::string flag(argv[i]);
		if(flag=="--F0" && i+1<argc) F0=atof(argv[++i]);
		else if(flag=="--recenter" && i+1<argc) recenter=atoi(argv[++i]);
		else if(flag=="--dump-every" && i+1<argc) dump_every=atoi(argv[++i]);
		else if(flag=="--row-offset" && i+1<argc) row_offset0=atol(argv[++i]);
		else if(flag=="--reflect-y") reflect_y=true;
		else{ std::cerr << "ERROR: unknown flag " << flag << std::endl; return 1; }
	}
	if(L%TILE!=0 || Ly%TILE!=0 || H0<=0 || recenter<TILE || 2*recenter+2*WALL_MARGIN>=Ly){
		std::cerr << "ERROR: need L, Ly multiples of " << TILE << ", H0 > 0, " << TILE
			<< " <= recenter < Ly/2 - " << WALL_MARGIN << std::endl;
		return 1;
	}
	if(H0*Ly/2+fabsf(h_u)>0.3f)
		std::cout << "WARNING: |h| reaches " << H0*Ly/2+fabsf(h_u) << " in the box (spinodal 0.385)" << std::endl;

	System sys(L,Ly,seed);
	// start flat at Ly/2; "escaped" = the wall reached WALL_MARGIN rows below Ly
	sys.set_geometry(Ly/2,Ly-WALL_MARGIN,false);
	sys.set_flat_interface();
	sys.set_row_offset(row_offset0);
	sys.set_reflecting_y(reflect_y);

	double area_to_u=1.0/L;
	long long area=sys.positive_area_lab();
	double u=area*area_to_u-0.5;
	double W=u+(F0-h_u)/H0;
	sys.set_external_field(h_u);

	std::cout << "Lx=" << L << " Ly=" << Ly << " H0=" << H0 << " h_u=" << h_u << " dW=" << dW
		<< " nsteps=" << nsteps << " seed=" << seed << " L_m=" << sqrt(2*sqrt(2.)/3/(2*H0)) << std::endl;

	char filename[100];
	sprintf(filename,"vd_seed%lu.dat",seed);
	std::ofstream out(filename);
	out << "# Lx=" << L << " Ly=" << Ly << " H0=" << H0 << " h_u=" << h_u << " dW=" << dW << " seed=" << seed
		<< " AMPDIS=" << AMPDIS << " AX=" << AX << " AY=" << AY << std::endl;
	out << "# step W u F S sweeps converged ell height Sdown n_mv depth" << std::endl;
	out.precision(10);

	for(long step=0;step<=nsteps;step++){
		if(step>0) W+=dW;
		sys.set_field_gradient(H0,float(W-sys.get_row_offset()));
		sys.save_state();
		int sweeps=sys.find_next_metastable(CHOP,true);
		if(sys.escaped()){
			std::cerr << "ERROR: step " << step << ": the wall reached the top of the box (Ly too small, or h_u/F0 too large)" << std::endl;
			return 1;
		}
		long long new_area=sys.positive_area_lab();
		u=new_area*area_to_u-0.5;
		double F=h_u+H0*(W-u);
		AvalancheGeometry g=sys.avalanche_geometry();
		out << step << " " << W << " " << u << " " << F << " " << new_area-area << " " << sweeps << " " << sys.converged()
			<< " " << g.ell << " " << g.height << " " << g.down << " " << g.n_mv << " " << g.depth_max << "\n";
		area=new_area;

		if(dump_every>0 && step%dump_every==0){
			sprintf(filename,"vdfield_step%ld_seed%lu.bin",step,seed);
			std::ofstream fout(filename,std::ios::binary);
			sys.print_field_binary(fout);
		}

		double above=u-sys.get_row_offset()-Ly/2;
		if(above>recenter) sys.shift_rows(TILE*int(above/TILE));
	}
	return 0;
}
