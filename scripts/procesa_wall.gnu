res; unset log xy

#file='wall.dat'
stats file nooutput;
nconfs=STATS_blocks-3;

print nconfs;

first=0;

if(modo==0){
plot for[i=0:nconfs] file index i smooth un w l t "".i
}



if(modo==1){
system("rm wall.dat.*.fft");
#do for[i=i0:nconfs:1]{
#do for[i=nconfs-i0:nconfs:1]{
do for[i=0:nconfs:1]{

	print "i=",i

	set table 'zzz'; 
	plot [][:] file u 1:2 index i smooth un; 
	unset table; 

	com=sprintf("gawk 'NR<%d && NF==3{print $2}' zzz > yyy",L); 
	system(com);


	stats 'yyy' nooutput
	
	if(STATS_max<0.95*L){ 
	fout=sprintf("%s.%d.fft",file,i);	
	com=sprintf("./fft1block.m yyy > %s",fout,i); 
	system(com);

	#fout=sprintf("%s.%d.bdr",file,i);	
	#com=sprintf("./bder.m yyy > %s",fout,i); 
	#system(com);

	#fout=sprintf("%s.%d.kwi",file,i);	
	#com=sprintf("./kwidth.m yyy > %s",fout,i); 
	#system(com);

	#fout=sprintf("%s.%d.kwi",file,i);	
	#com=sprintf("gnuplot -e \"infile='yyy'\" -e \"outfile=%s\" detrendedwidth.gnu",fout,i); 
	#system(com);

	#print i;
	}
}
}

#p for[i=1:nconfs] (fout=sprintf("%s.%d.fft",file,i)) u (log($1)):(log($2)) w lp t sprintf("%d",i),\
#-x*(1+2*1.25), -x*(1+2*0.63);

# proceso todos los seeds
#do for[seed in "1 3 4 5"]{file=sprintf("wall_%d.dat",0+seed); modo=1; load "extrae_Sq_from_wall.gnu"; print file;};

# grafico el promedio
# p for[i=1:19] sprintf("< cat wall_*.dat.%d.fft | awk 'NF==2'",i) u 1:2 smooth un w lp


