%Copyright (c) 2013   Ramon Franquesa Alberti, Carlos Martï¿œn Isla , Gonzalo Lopez Lillo , Aleix Gras Godoy 


clear all;
%%
types=struct;
types(1).type='concert';
types(2).type='conference';
types(3).type='exhibition';
types(4).type='fashion';
types(5).type='non_event';
types(6).type='other';
types(7).type='protest';
types(8).type='sports';
types(9).type='theater_dance';
ndeclasses=length(types);
flag=3;
tipusExtraccio='HISTBLOC';



%% LECTURA D'IMATGES, EXTRACCIï¿œ Y ESCRIPTURA DE MODELS

% saltar este paso si se dispone de modelosX.bin
tic
elementsperclasse=[];
dataset=[];
fid=fopen(strcat('modelos',tipusExtraccio,'.bin'),'w');
fwrite(fid,[]);
fclose(fid);
fid=fopen(strcat('modelos',tipusExtraccio,'.bin'),'a+');


for k=1:ndeclasses,
    
directori=strcat(types(k).type,'\');
model=lecturaImatges(directori,flag);

aux=dataset;
dataset=[aux ; model];

[m n]=size(model);
aux=elementsperclasse;
elementsperclasse=[aux  m];



end;

fwrite(fid,elementsperclasse,'uint16'); % PRIMERA FILA DE L'ARXIU : VECTOR AMB SEPARACIONS DE LA MATRIU DATASET
fwrite(fid,dataset, 'double'); % MATRIU AMB TOTS ELS MODELS, QUE ES DESENSAMBLA AMB EL VECTOR ANTERIOR
fclose(fid);
%%D'aquesta forma emns estalviem fer servir structs per models de mida variable, i aprofitem tota la
%%potï¿œncia de matlab.
toc
%% RECUPERACIï¿œ DE MODELS


fid=fopen(strcat('modelos',tipusExtraccio,'.bin'),'r');
index=fread(fid,[1 ndeclasses],'uint16');
    
auxmat=fread(fid,[sum(index) 256*16],'double'); 




%% CLASSIFICACIï¿œ


archivo=fopen(strcat('resultats',tipusExtraccio,'.txt'),'w');
directori='clas/';
[nombre,n_model,H]=lecturaImatges_Aval(directori, flag);
[nil n]=size(n_model);%nil=numero de imagenes leidas
k=1;
for i=1:nil
    indice=classificador_knn(n_model(i,:),auxmat,index,k);
    types(indice).type;

    fprintf(archivo,strcat(nombre(i).noms));
    fprintf(archivo,' ');
    fprintf(archivo,types(indice).type);
    fprintf(archivo,'\r\n');
    
      if(mod(i,100)==0) %progreso
        x=num2str(floor(i*100/nil));
        display(strcat(x,'%'));end;
end
fclose(archivo);



%% AVALUACIï¿œ
%%importar csv

close all
clear all
fid= fopen('sed2013_task2_dataset_train_gs.csv');
C = textscan(fid, '%s%s');
fclose(fid);

L=length(C{1});
k=1;

id=C{1}(2:L);
l=length(unique(id));
clear id;
tag=cell(l,1);
Id = tag;
ref = char(C{1}(2));
stri= '';
for i=1:L-1                 % map key: id_photo data: tags
    len = length(char(C{1}(i+1)));
    if len == length (ref)
        if sum (char(C{1}(i+1))== ref) == len 
            stri=[stri ' ' char(C{2}(i+1))];
        else
            tag(k) = cellstr(stri);
            Id(k) = C{1}(i) ;
            stri = '';
            stri=[stri ' ' char(C{2}(i+1))];
            k = k+1;
        end
    else
        tag(k) = cellstr(stri);
        Id(k) = C{1}(i) ;
        stri = '';
        stri=[stri ' ' char(C{2}(i+1))];
        k = k+1;
            
    end
    ref = char(C{1}(i+1));
    
end
    tag(k) = cellstr(stri);
    Id(k) = C{1}(i) ;


map_tag = containers.Map(Id,tag);
%
clear Id tag C stri len ref i l fid k;


% LECTURA CLASSES

fid= fopen('sed2013_task2_dataset_train_gs.csv');
M = textscan(fid, '%s%s');
fclose(fid);
L = length(M{1});
id = M {1}(2:L);
class = M {2}(2:L);



data_map = containers.Map(id,class);    %map key: id_photo data: class

clear data L M fid;

%

IN=[1 2 3 4 5 6 7 8 9];
OUT1=[NaN NaN NaN NaN NaN NaN NaN NaN NaN];
ext = dir('11.txt'); %extencio imatges // INTRODUIR ARXIU A CLASSIFICAR!!!!!//
W = numel(ext);
for i = 1:W    %loop lectura

filename = ext(i).name;
fid= fopen(filename);
M = textscan(fid, '%s%s');
fclose(fid);
l = length(M{1});
    N = M{1};
    C = M{2};
    
   OUT =[N C];



result = zeros (9);                %Matriu de confusio
e = 0;
true = 0;

for j = 1 : length(OUT)    %length(R)

    N = char(OUT (j,1));
    C = char(OUT (j,2));
  
    
    p = classidentify(C);         %classe predicció

    if isKey(data_map,N)
        gt = classidentify (data_map(N)); %classe ground truth
    end
    
      IN(j+9) = p;
     OUT1(j+9) = gt;
    result(gt,p) = result(gt,p)+1;
end
end

%


g2 = [IN];
g1 = [OUT1];
[C,order] = confusionmat(g1,g2);
Pr=0;
Re=0;
Fscore=0;

%Avaluacio de les dades
for l = 1:9
PV=C(l,l);
PF=0;
NF=0;
for j = 1:9
   PF=PF+C(l,j);
   NF=NF+C(j,l);
end
  PF=PF-C(l,l);
  NF=NF-C(l,l);
Pr(l) = (PV/(PV+PF));
Re(l) = (PV/(PV+NF));
Fscore(l) = 2*((Pr(l)*Re(l))/(Pr(l)+Re(l)));

end

% Calculem quants ground truth hi ha.

KKOUT=OUT1;
c1=0;c2=0;c3=0;c4=0;c5=0;c6=0;c7=0;c8=0;c9=0;
kout= length(KKOUT);
for pout = 9:kout;
    
   if( KKOUT(pout) == 1)
       c1=1;    
   end
   if( KKOUT(pout) == 2)
       c2=1;    
   end
    if( KKOUT(pout) == 3)
       c3=1;    
    end
    if( KKOUT(pout) == 4)
       c4=1;    
    end
    if( KKOUT(pout) == 5)
       c5=1;    
    end
    if( KKOUT(pout) == 6)
       c6=1;    
    end
    if( KKOUT(pout) == 7)
       c7=1;    
    end
    if( KKOUT(pout) == 8)
       c8=1;    
    end
    if( KKOUT(pout) == 9)
       c9=1;    
   end

end

ctotal = c1+c2+c3+c4+c5+c6+c7+c8+c9;
% Calcul de la precisió total!!
Fstotal=0;
Puta=isnan(Fscore);
Fscore;
for pres = 1:length(Fscore)
   
    if(Puta(pres) == 0)
        Fstotal=Fstotal+Fscore(pres);
    
    end
  
end
Fstotal=Fstotal/ctotal
Re_avg = mean(Re);
Pr_avg = mean(Pr);
%
kk = length(IN);
Encerts= 0;
Fals= 0;
for KK = 9:kk

if (IN(KK) == OUT1(KK))
   Encerts= Encerts+1;
else Fals = Fals +1;
end
end

Accuraciy = (Encerts/(Encerts+Fals))*100;


%
f = figure('Position',[400 400 800 500]);
dat = C; 
cnames = {'Concert','Conference','Exhibition','Fashion','Non_event','Others','Protest','Sports','Theatre'};
rnames = {'Concert','Conference','Exhibition','Fashion','Non_event','Others','Protest','Sports','Theatre'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 200 700 200 ]);
set(t,'ColumnWidth',{50})

dat = [Pr]; 
rnames = {'Precision'};
cnames = {'Concert','Conference','Exhibition','Fashion','Non_event','Others','Protest','Sports','Theatre'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 0 700 50]);
set(t,'ColumnWidth',{50})

dat = [Re]; 
rnames = {'Recall'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 50 700 50]);
set(t,'ColumnWidth',{50})

dat = [Fscore];
rnames = {'F-score'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 100 700 50]);
set(t,'ColumnWidth',{50})

dat = [Pr_avg Re_avg Fstotal Accuraciy];
rnames = {'Avg'};
cnames = {'Precision', 'Recall','F-score', 'Accuracity'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 150 300 50]);
set(t,'ColumnWidth',{50})
%%
figure(3);
subplot(4,1,1);
plot(Pr);title('Precision');
axis([1 9 0 1]);
subplot(4,1,2);
plot(Re);title('Recall');
axis([1 9 0 1]);
subplot(4,1,3);
plot(Fscore);title('F-score');
axis([1 9 0 1]);
subplot(4,1,4);
plot(Pr,Re);title('Corba PR');ylabel('Precision');xlabel('Recall');
axis([0 1 0 1]);
