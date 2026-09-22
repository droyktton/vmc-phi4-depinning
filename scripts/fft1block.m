#!/usr/bin/octave -qf

arg_list = argv();
sample=load(arg_list{1}); 
L=size(sample)(1);

u=sample(:,1);

;s=abs(fft(u).^2);
s=abs(fft(u)).^2;

r=[1:L/2]; 
q=[0:L/2-1]*2*pi/L;
ss=s(r); 

disp([q',ss])
#disp(std(u))


#fprintf("%1.2f %1.2f\n', [q',ss]")
