for f in $1/critica*dat
do 
	./tiene_overhang.awk $f; 
done | awk '{if($1>0) {n++}; frac=frac+$1;m++;}END{print n/m, frac/m, m}'
