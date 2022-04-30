function [power,phase]=dataout(filein,fileout,line1,length)
%Read given GRAPE pulse and return the power and phase
% form: [power,phase]=dataout(filein,fileout,line1,length)
% need to give the name of the GRAPE, the output file name, the starting line (usually 20), and the number of points.

fidin=fopen(filein,'rt');
fidout=fopen(fileout,'wt');
nline=0;
line2 = line1+length-1;

while ~feof(fidin) %
tline=fgetl(fidin); %
nline=nline+1;
if nline==line1
    tline = strrep(tline,',',' ');
    fprintf(fidout,'%s\n',tline);
if line1==line2
else line1=line1+1;
end
end
end


[power,phase] = textread(fileout,'%f%f');

fclose(fidin);
fclose(fidout);

