for size in 128 256 512 1024 2048 4096
do
	for dis in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
		do echo $dis $(./overhangprob.sh L$size\_$dis); 
	done > overhangs$size.dat
	echo $size "listo"
done
