#ejemplo de uso
#gnuplot -e "L=512; file='critica_h0.007800.dat'" extrae_Sq.gnu

res; unset log xy

stats file;
print "file=".file

comando=sprintf("gawk -f tiene_overhang.awk %s | awk '{if($1==0) print \"SIN\"; else print \"CON\"}'",file)
clase=system(comando)

set table 'zzz'; 
plot [][:] file u 1:2 smooth un; 
unset table; 

comando=sprintf("gawk 'NR<%d && NF==3{print $2}' zzz > yyy",L); 
system(comando)


fout=sprintf("%s.%s.fft",file,clase);	
com=sprintf("./fft1block.m yyy > %s",fout); 
system(com);

stats 'yyy'
wout=sprintf("%s.%s.w",file,clase);	
set print wout
print STATS_stddev
set print

#p for[i=1:nconfs] (fout=sprintf("%s.%d.fft",file,i)) u (log($1)):(log($2)) w lp t sprintf("%d",i),\
#-x*(1+2*1.25), -x*(1+2*0.63);

# proceso todos los seeds
#do for[seed in "1 3 4 5"]{file=sprintf("wall_%d.dat",0+seed); modo=1; load "extrae_Sq_from_wall.gnu"; print file;};

# grafico el promedio
# p for[i=1:19] sprintf("< cat wall_*.dat.%d.fft | awk 'NF==2'",i) u 1:2 smooth un w lp

print "======================"
#print "EJEMPLO:"
#print "gnuplot -e \"file=\'critica_h0.007800.dat\'\" extrae_Sq.gnu"
print "SALIDA:"
print fout
print wout
print "======================"

