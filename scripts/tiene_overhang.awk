#! /usr/bin/awk -f 
BEGIN{
	L=0;
	y2=0;
	y=0;
	N=0;
}
{
	if($1!="#"){
		n[$1]++;
		if($1>L) L=$1;
		y2+=$2*$2;
		y+=$2;
		N++;
	};
}
END{
	m=0;
	for(i in n){
		if(max<n[i]) max=n[i];
		if(n[i]>=6) m++;
	} 

	#if(max>=6) print "si tiene overhangs =", m; 
	#else print "no tiene overhangs";

	print m/L, y2/N-y*y/(N*N);
}

