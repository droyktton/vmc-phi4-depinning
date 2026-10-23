// Velocity-driven ensemble at finite velocity for the disordered phi^4
// model: the real dynamics, integrated with explicit Euler at T = 0.
//
// The same spring as main_vd.cu, h(y,t) = h_u - H0*(y - W(t)), but the
// well center now advances continuously, W(t) = W0 + v*t, and the field
// evolves with
//   dphi/dt = c*Laplacian(phi) + eps0*[(1+r)*phi - phi^3] + h(y,t)
// (System::euler_step, the force of velocityop). The wall then moves at
// mean velocity v, and the drive F = h_u + H0*(W - u) it needs gives the
// force-velocity curve F(v), its fluctuations in time, and for v -> 0
// the quasi-static result of main_vd.cu. Moving the disorder backwards,
// r(x, y + v t), is the same as moving the well; the conveyor belt
// (System::shift_rows) keeps the wall in the box by whole rows, an exact
// symmetry, so the disorder never needs interpolating.
//
// Output, vdyn_v<v>_seed<seed>.dat, every tout time units:
//   t W u F
//
// Fixed-force mode (H0 = 0): no spring, a uniform field h = h_u, and the
// wall moves at whatever velocity it finds; v is ignored and the output,
// vdyn_h<h_u>_seed<seed>.dat, is t u (the velocity is the slope of u(t)).
// Comparing (h, v) from this mode with (<F>, v) from the velocity-driven
// one tests the equivalence of the two ensembles at finite v.
// u = (area of the phi>0 phase)/Lx - 1/2, lab frame, as in main_vd.cu.
// With --dump-every k, the field every k outputs as raw float32
// (vdynfield_t<t>_seed<seed>.bin, Lx*Ly, box frame).
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <cmath>
#include "misistema.h"

// the wall may come no closer than this to the top of the box
#define WALL_MARGIN 32

int main(int argc, char **argv)
{
	if(argc<9){
		std::cout << "usage: " << argv[0] << " L Ly H0 h_u v dt tmax seed [--tout t] [--F0 f] [--recenter d] [--row-offset R] [--dump-every k] [--start-from file]" << std::endl;
		std::cout << "  L          lattice size Lx along the wall (multiple of 32)" << std::endl;
		std::cout << "  Ly         box height along the motion (multiple of 32)" << std::endl;
		std::cout << "  H0         field gradient (spring 2*H0, mass length sqrt(0.47/H0)); 0: fixed force h_u" << std::endl;
		std::cout << "  h_u        uniform field (about the expected drive)" << std::endl;
		std::cout << "  v          velocity of the well center, rows per unit time" << std::endl;
		std::cout << "  dt         Euler time step (stable below ~0.19; 0.05 is safe)" << std::endl;
		std::cout << "  tmax       total time" << std::endl;
		std::cout << "  seed       disorder-realization seed" << std::endl;
		std::cout << "  --tout t       output interval (default 1)" << std::endl;
		std::cout << "  --F0 f         initial drive (default h_u): W0 = u + (f-h_u)/H0" << std::endl;
		std::cout << "  --recenter d   shift the box once the wall is more than d rows above Ly/2 (default 64)" << std::endl;
		std::cout << "  --row-offset R the box starts at lab row R (default 0)" << std::endl;
		std::cout << "  --dump-every k save the field every k outputs" << std::endl;
		std::cout << "  --start-from f start from a field saved by main_vd.cu/main_vdyn.cu (same L, Ly; give its --row-offset)" << std::endl;
		return 1;
	}
	int L=atoi(argv[1]), Ly=atoi(argv[2]);
	float H0=atof(argv[3]), h_u=atof(argv[4]);
	double v=atof(argv[5]);
	float dt=atof(argv[6]);
	double tmax=atof(argv[7]);
	unsigned long seed=strtoul(argv[8],NULL,10);
	float F0=h_u;
	double tout=1.0;
	int recenter=64, dump_every=0;
	long row_offset0=0;
	std::string start_from;
	for(int i=9;i<argc;i++){
		std::string flag(argv[i]);
		if(flag=="--tout" && i+1<argc) tout=atof(argv[++i]);
		else if(flag=="--F0" && i+1<argc) F0=atof(argv[++i]);
		else if(flag=="--recenter" && i+1<argc) recenter=atoi(argv[++i]);
		else if(flag=="--row-offset" && i+1<argc) row_offset0=atol(argv[++i]);
		else if(flag=="--dump-every" && i+1<argc) dump_every=atoi(argv[++i]);
		else if(flag=="--start-from" && i+1<argc) start_from=argv[++i];
		else{ std::cerr << "ERROR: unknown flag " << flag << std::endl; return 1; }
	}
	bool fixed_force=(H0==0);
	if(L%TILE!=0 || Ly%TILE!=0 || H0<0 || dt<=0 || recenter<TILE || 2*recenter+2*WALL_MARGIN>=Ly){
		std::cerr << "ERROR: need L, Ly multiples of " << TILE << ", H0 >= 0, dt > 0, " << TILE
			<< " <= recenter < Ly/2 - " << WALL_MARGIN << std::endl;
		return 1;
	}
	if(dt>0.15f) std::cout << "WARNING: dt = " << dt << " is close to the Euler stability limit (~0.19)" << std::endl;
	if(H0*Ly/2+fabsf(h_u)>0.3f)
		std::cout << "WARNING: |h| reaches " << H0*Ly/2+fabsf(h_u) << " in the box (spinodal 0.385)" << std::endl;

	System sys(L,Ly,seed);
	// "escaped" = the wall reached WALL_MARGIN rows below Ly
	sys.set_geometry(Ly/2,Ly-WALL_MARGIN,false);
	if(start_from.empty()) sys.set_flat_interface();
	else{
		std::ifstream fin(start_from.c_str(),std::ios::binary);
		if(!fin){ std::cerr << "ERROR: cannot read " << start_from << std::endl; return 1; }
		sys.read_field_binary(fin);
	}
	sys.set_row_offset(row_offset0);
	sys.set_external_field(h_u);

	if(fixed_force && (v!=0 || F0!=h_u)) std::cout << "WARNING: H0 = 0 (fixed force h = h_u): v and --F0 are ignored" << std::endl;
	if(fixed_force) v=0;

	double area_to_u=1.0/L;
	double u=sys.positive_area_lab()*area_to_u-0.5;
	double W0=fixed_force ? 0 : u+(F0-h_u)/H0;

	long nsteps=lround(tmax/dt), nout=std::max(1L,lround(tout/dt));
	std::cout << "Lx=" << L << " Ly=" << Ly << " H0=" << H0 << " h_u=" << h_u << " v=" << v << " dt=" << dt
		<< " tmax=" << tmax << " seed=" << seed;
	if(fixed_force) std::cout << " (fixed force)" << std::endl;
	else std::cout << " L_m=" << sqrt(2*sqrt(2.)/3/(2*H0)) << std::endl;

	char filename[200];
	if(fixed_force) sprintf(filename,"vdyn_h%g_seed%lu.dat",h_u,seed);
	else sprintf(filename,"vdyn_v%g_seed%lu.dat",v,seed);
	std::ofstream out(filename);
	out << "# Lx=" << L << " Ly=" << Ly << " H0=" << H0 << " h_u=" << h_u << " v=" << v << " dt=" << dt
		<< " tout=" << nout*dt << " seed=" << seed << " AMPDIS=" << AMPDIS << " AX=" << AX << " AY=" << AY << std::endl;
	out << (fixed_force ? "# t u" : "# t W u F") << std::endl;
	out.precision(10);

	for(long step=0;step<=nsteps;step++){
		double t=step*double(dt);
		double W=W0+v*t;
		if(step%nout==0){
			u=sys.positive_area_lab()*area_to_u-0.5;
			if(fixed_force) out << t << " " << u << "\n";
			else out << t << " " << W << " " << u << " " << h_u+H0*(W-u) << "\n";
			if(sys.escaped()){
				std::cerr << "ERROR: t=" << t << ": the wall reached the top of the box (Ly too small, or v, h_u too large)" << std::endl;
				return 1;
			}
			long iout=step/nout;
			if(dump_every>0 && iout%dump_every==0){
				sprintf(filename,"vdynfield_t%g_seed%lu.bin",t,seed);
				std::ofstream fout(filename,std::ios::binary);
				sys.print_field_binary(fout);
			}
			double above=u-sys.get_row_offset()-Ly/2;
			if(above>recenter) sys.shift_rows(TILE*int(above/TILE));
		}
		if(step==nsteps) break;
		sys.set_field_gradient(H0,float(W-sys.get_row_offset()));
		sys.euler_step(dt);
	}
	return 0;
}
