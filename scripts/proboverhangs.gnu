set multi lay 1,2; 

psize=1


#set log y

set key left
set yla 'Prob overhangs'; set xla 'Desorden'; 
plot [][0:1] for[size in '128 256 512 1024 2048 4096'] 'overhangs'.size.'.dat' u ($1):2:(1./$4) w errorl t 'L='.size ps psize; 

unset key
set xla 'Desorden L^{0.45}';
plot [:40][0:1] for[size in '128 256 512 1024 2048 4096'] 'overhangs'.size.'.dat' u ($1*size**a):2:(1/$4) w error t 'L='.size ps psize,\
(1+tanh((x-17)/5))/2.0 lt 0


#set yla 'overhang fraction'; set xla 'Desorden'; 
#plot for[size in '128 256 512 1024 2048 4096'] 'overhangs'.size.'.dat' u ($1):3 w lp t 'L='.size; 

#set xla 'Desorden L^{0.45}';
#plot for[size in '128 256 512 1024 2048 4096'] 'overhangs'.size.'.dat' u ($1*size**0.45):3 w lp t 'L='.size

unset multi

