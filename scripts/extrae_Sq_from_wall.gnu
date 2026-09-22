res; unset log xy

#file='wall.dat'
stats file;
nconfs=STATS_blocks-2;

first=0;
modo=1;

if(modo==1){
do for[i=1:nconfs:1]{
	set table 'zzz'; 
	plot [][:] file u 1:2 index i smooth un; 
	unset table; 

	system("gawk 'NR<512 && NF==3{print $2}' zzz > yyy"); 

	fout=sprintf("%s.%d.fft",file,i);	
	com=sprintf("./fft1block.m yyy > %s",fout,i); 
	system(com);

	print i;
}
#modo=0;
}

#p for[i=1:nconfs] (fout=sprintf("%s.%d.fft",file,i)) u (log($1)):(log($2)) w lp t sprintf("%d",i),\
#-x*(1+2*1.25), -x*(1+2*0.63);

# proceso todos los seeds
#do for[seed in "1 3 4 5"]{file=sprintf("wall_%d.dat",0+seed); modo=1; load "extrae_Sq_from_wall.gnu"; print file;};

# grafico el promedio
# p for[i=1:19] sprintf("< cat wall_*.dat.%d.fft | awk 'NF==2'",i) u 1:2 smooth un w lp


