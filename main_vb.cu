// Quasi-static bubble (v -> 0+) for the disordered phi^4 model, with the
// VMC solver of misistema.h: the radial version of main_vd.cu.
//
// A disk of the +1 phase, radius R0, sits in the centre of an L x L box,
// periodic in x and y (System::set_bubble). The field is a cone,
// h(r) = h_u - H0*(r - R_W), r the distance from the centre of the box,
// so a bubble of radius R feels the drive F = h_u + H0*(R_W - R): the
// spring of main_vd.cu, now acting along the normal of a closed wall,
// with the same mass length L_m = sqrt(sigma/2H0). The disorder stays
// put; at every step the well radius R_W advances by dA/(2 pi R), so that
// each step adds about the same area dA whatever the perimeter, and the
// bubble relaxes to the next metastable state. F settles at about
// h_d + sigma/2R, the threshold plus the Laplace pressure of the
// curvature (sigma = 2 sqrt(2)/3 for c = eps0 = 1). This is stable only
// for R > L_m (H0 > sigma/2R^2); start with R0 > L_m and F0 a little above
// sigma/2R0, or the bubble collapses (a curved wall feels no
// Peierls-Nabarro pinning). The run ends when the
// bubble reaches the escape radius Re, i.e. before it feels its periodic
// images. This is the T = 0 counterpart of a bubble domain expanding by
// creep in a film (Durin et al., PRB 110, L020405 (2024)).
//
// Output, vb_seed<seed>.dat, one line per step:
//   step R_W R F S sweeps converged ncomp
// R = sqrt(A/pi), A the area of the phi>0 phase; S = change of A over the
// step (sites); ncomp = connected clusters of flipped sites.
// vbaval_seed<seed>.dat, one line per cluster (System::avalanche_components):
//   step S Sdown xc yc cxx cyy cxy span_x span_y r theta ell_theta dr
// S = up - down sites, Sdown = sites that went back to phi<=0 (0 if
// no-passing holds); centroid, covariance (principal axes: the
// longitudinal length of Durin et al.), extents in x and y; polar
// coordinates of the centroid about the centre; ell_theta = angular
// extent times r (along the wall) and dr = radial extent (across it).
// vbflip_seed<seed>.bin, int32, L*L: for each site, the step at which it
// last became phi>0 (-1: inside the initial disk, before step 0;
// INT_MAX: outside at the end, or flipped only in the cut last step).
// The bubble at step t is {flip <= t}, which gives the wall R(theta)
// at every step, and the space-time clustering of Durin et al.
// With --dump-every k, the field every k steps as raw float32
// (vbfield_step<step>_seed<seed>.bin, L*L).
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <cmath>
#include "misistema.h"

// relaxation is checked every CHOP sweeps
#define CHOP 1000

int main(int argc, char **argv)
{
	if(argc<7){
		std::cout << "usage: " << argv[0] << " L H0 h_u dA nsteps seed [--R0 r] [--F0 f] [--Re re] [--dump-every k]" << std::endl;
		std::cout << "  L          box size, L x L (multiple of 32)" << std::endl;
		std::cout << "  H0         radial field gradient (spring 2*H0, mass length sqrt(0.47/H0))" << std::endl;
		std::cout << "  h_u        uniform field (about the expected h_d)" << std::endl;
		std::cout << "  dA         area added per step: R_W advances by dA/(2 pi R)" << std::endl;
		std::cout << "  nsteps     largest number of steps (the run stops earlier at Re)" << std::endl;
		std::cout << "  seed       disorder-realization seed" << std::endl;
		std::cout << "  --R0 r     initial radius (default 50)" << std::endl;
		std::cout << "  --F0 f     initial drive (default h_u): R_W = R0 + (f-h_u)/H0" << std::endl;
		std::cout << "  --Re re    escape radius: stop once phi>0 beyond it (default L/2 - 64)" << std::endl;
		std::cout << "  --dump-every k save the field every k steps" << std::endl;
		return 1;
	}
	int L=atoi(argv[1]);
	float H0=atof(argv[2]), h_u=atof(argv[3]);
	double dA=atof(argv[4]);
	long nsteps=atol(argv[5]);
	unsigned long seed=strtoul(argv[6],NULL,10);
	float R_init=50, F0=h_u, Re=L/2-64;
	int dump_every=0;
	for(int i=7;i<argc;i++){
		std::string flag(argv[i]);
		if(flag=="--R0" && i+1<argc) R_init=atof(argv[++i]);
		else if(flag=="--F0" && i+1<argc) F0=atof(argv[++i]);
		else if(flag=="--Re" && i+1<argc) Re=atof(argv[++i]);
		else if(flag=="--dump-every" && i+1<argc) dump_every=atoi(argv[++i]);
		else{ std::cerr << "ERROR: unknown flag " << flag << std::endl; return 1; }
	}
	if(L%TILE!=0 || H0<=0 || R_init<=0 || Re<=R_init || Re>=L/2){
		std::cerr << "ERROR: need L a multiple of " << TILE << ", H0 > 0, 0 < R0 < Re < L/2" << std::endl;
		return 1;
	}

	System sys(L,L,seed);
	sys.set_bubble(R_init,Re);
	sys.set_flat_interface(); // the disk
	sys.set_external_field(h_u);

	const double sigma=2*sqrt(2.)/3;
	long long area=sys.positive_area_lab();
	double R=sqrt(area/M_PI);
	double W=R+(F0-h_u)/H0;
	// the spring holds the bubble only if it is stiffer than the Laplace
	// pressure, H0 > sigma/2R^2, i.e. R > L_m: a smaller one collapses or
	// runs away
	if(R<sqrt(sigma/(2*H0)))
		std::cout << "WARNING: R0 < L_m = " << sqrt(sigma/(2*H0)) << ": the bubble is unstable until it grows past L_m" << std::endl;
	if(H0*W+h_u>0.3f || h_u-H0*(L/sqrt(2.)-W)<-0.3f)
		std::cout << "WARNING: the cone reaches |h| > 0.3 in the box (spinodal 0.385)" << std::endl;

	std::cout << "L=" << L << " H0=" << H0 << " h_u=" << h_u << " dA=" << dA << " nsteps=" << nsteps << " seed=" << seed
		<< " R0=" << R_init << " Re=" << Re << " L_m=" << sqrt(sigma/(2*H0)) << std::endl;

	char filename[100];
	sprintf(filename,"vb_seed%lu.dat",seed);
	std::ofstream out(filename);
	sprintf(filename,"vbaval_seed%lu.dat",seed);
	std::ofstream aval(filename);
	out << "# L=" << L << " H0=" << H0 << " h_u=" << h_u << " dA=" << dA << " seed=" << seed << " R0=" << R_init << " Re=" << Re
		<< " AMPDIS=" << AMPDIS << " AX=" << AX << " AY=" << AY << std::endl;
	out << "# step R_W R F S sweeps converged ncomp" << std::endl;
	aval << "# step S Sdown xc yc cxx cyy cxy span_x span_y r theta ell_theta dr" << std::endl;
	out.precision(10); aval.precision(8);

	const int NEVER=0x7fffffff;
	std::vector<int> flip(size_t(L)*L,NEVER);
	{
		std::vector<char> m=sys.positive_mask();
		for(size_t i=0;i<m.size();i++) if(m[i]) flip[i]=-1;
	}

	for(long step=0;step<=nsteps;step++){
		if(step>0) W+=dA/(2*M_PI*R);
		sys.set_radial_field_gradient(H0,float(W));
		sys.save_state();
		int sweeps=sys.find_next_metastable(CHOP,true);
		if(sys.collapsed()){
			std::cerr << "ERROR: step " << step << ": the bubble collapsed (F0 too low)" << std::endl;
			return 1;
		}
		bool done=sys.escaped();
		long long new_area=sys.positive_area_lab();
		R=sqrt(new_area/M_PI);
		double F=h_u+H0*(W-R);
		std::vector<AvalancheComponent> comps=sys.avalanche_components();
		out << step << " " << W << " " << R << " " << F << " " << new_area-area << " " << sweeps << " "
			<< (done ? -1 : int(sys.converged())) << " " << comps.size() << "\n";
		for(const AvalancheComponent &c: comps)
			aval << step << " " << c.up-c.down << " " << c.down << " " << c.xc << " " << c.yc << " "
				<< c.cxx << " " << c.cyy << " " << c.cxy << " " << c.span_x << " " << c.span_y << " "
				<< c.r << " " << c.theta << " " << c.ell_theta << " " << c.dr << "\n";
		area=new_area;
		if(!done)
			for(size_t k=0;k<sys.flipped_idx.size();k++)
				flip[sys.flipped_idx[k]] = sys.flipped_phi[k]>0 ? int(step) : NEVER;

		if(dump_every>0 && (step%dump_every==0 || done)){
			sprintf(filename,"vbfield_step%ld_seed%lu.bin",step,seed);
			std::ofstream fout(filename,std::ios::binary);
			sys.print_field_binary(fout);
		}
		if(done || step==nsteps){
			sprintf(filename,"vbflip_seed%lu.bin",seed);
			std::ofstream fout(filename,std::ios::binary);
			fout.write(reinterpret_cast<const char*>(flip.data()),flip.size()*sizeof(int));
		}
		if(done){
			std::cout << "step " << step << ": the bubble reached the escape radius " << Re << " (R = " << R << "); the last step is cut (converged = -1)" << std::endl;
			break;
		}
	}
	return 0;
}
