# dado un L y u D, genera por cada critica del directorio correspondiente
# un archivo con el mismo nombre pero extension .fft, para hacer luego los promedios

L=$1
D=$2

for f in L$L\_$D/critica_*.dat
do 
	echo $f; 
	gnuplot -e "L=$L; file='$f'" extrae_Sq.gnu ; 
done

#visualizar para un tamanio y desorden
#set table; set out 'zzz'; plot for[file in system('ls L4096_0.1/*fft')] file u 1:2 w lp; unset table; p "< sed '/^$/d' zzz" smooth un, 1./x**(1+2*1.25)
