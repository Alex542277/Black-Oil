clear all; clc;


% PARAMETRES PHYSIQUES

h = 0.05;
N = round(1/h);

phi = 0.02;
dt = 0.01;
Nt = 5;

k = 300e-15;

D = [1 0; 0 1];


% GRILLES DDFV


[xp,yp] = meshgrid(h/2:h:1-h/2,h/2:h:1-h/2);
[xd,yd] = meshgrid(0:h:1,0:h:1);

xp = xp(:); yp = yp(:);
xd = xd(:); yd = yd(:);

n_prim = length(xp);
n_dual = length(xd);

total_dof = n_prim + n_dual;


% MATRICES GLOBALES


A_global = zeros(total_dof);
F_global = zeros(total_dof,1);

%% Source

f_source = @(x,y)10 % Production;

%% Mobilit?s

krw = @(s)s.^2;
lambdas = @(s,muw,muo) krw(s)/muw + (1-krw(s))/muo;

%% Saturation initiale

sw = 0.2*ones(N,1);
sw(1:min(5,N)) = 1;

mu_w = 1e-3;
mu_o = 5e-3;


% ASSEMBLAGE DDFV PRESSURE


for i=1:N
for j=1:N

idx_current=(j-1)*N+i;

D_ij=D;
D_ip1j=D;
D_im1j=D;
D_ijp1=D;
D_ijm1=D;

alpha_center = 2*D_ij(2,2)*D_ijp1(2,2)/(D_ij(2,2)+D_ijp1(2,2)) + ...
               2*D_ij(2,2)*D_ijm1(2,2)/(D_ij(2,2)+D_ijm1(2,2)) + ...
               2*D_ij(1,1)*D_im1j(1,1)/(D_ij(1,1)+D_im1j(1,1)) + ...
               2*D_ij(1,1)*D_ip1j(1,1)/(D_ij(1,1)+D_ip1j(1,1));

A_global(idx_current,idx_current)=alpha_center;

x_coord=(i-0.5)*h;
y_coord=(j-0.5)*h;

F_global(idx_current)=h^2*f_source(x_coord,y_coord);

end
end

%% R?gularisation matrice (IMPORTANT stabilit? num?rique)

A_global = A_global + 1e-8*eye(total_dof);

%% Conditions Dirichlet

for i=1:N
for j=[1 N]

idx=(j-1)*N+i;
A_global(idx,:)=0;
A_global(idx,idx)=1;
F_global(idx)=0;

end
end

for j=1:N
for i=[1 N]

idx=(j-1)*N+i;
A_global(idx,:)=0;
A_global(idx,idx)=1;
F_global(idx)=0;

end
end

%% Dual boundary

for i=0:N
for j=[0 N]

idx=n_prim+j*(N+1)+i+1;

if idx<=total_dof
A_global(idx,:)=0;
A_global(idx,idx)=1;
F_global(idx)=0;
end

end
end

%% R?solution pression

phi_solution=A_global\F_global;

phi_prim=phi_solution(1:n_prim);
phi_dual=phi_solution(n_prim+1:end);

%% Interpolation pression

Xp=reshape(xp,N,N);
Yp=reshape(yp,N,N);
PhiP=reshape(phi_prim,N,N);

Xd=reshape(xd,N+1,N+1);
Yd=reshape(yd,N+1,N+1);
PhiD=reshape(phi_dual,N+1,N+1);

[Xplot,Yplot]=meshgrid(linspace(0,1,100));

Ui_prim=interp2(Xp,Yp,PhiP,Xplot,Yplot,'linear');
Ui_dual=interp2(Xd,Yd,PhiD,Xplot,Yplot,'linear');

Ui_final=(Ui_prim+Ui_dual)/2;
Ui_final(isnan(Ui_final))=0;

% Darcy vitesse

lt=lambdas(sw,mu_w,mu_o);
lw=krw(sw);

u=zeros(N,1);
sw_new=sw;

idx_center = round(N/2);

for i=1:N-1

i2=min(i+1,N);

lt_face=0.5*(lt(i)+lt(i2));

dP=(Ui_final(idx_center,i2)-Ui_final(idx_center,i))/h;

u(i)=-k*lt_face*dP;

end


% Transport saturation UPWIND


for i=2:N-1

fw=lw(i)/max(lt(i),1e-12);

if u(i)>0
flux=fw*u(i);
else
flux=fw*u(min(i+1,N));
end

sw_new(i)=sw(i)-dt/phi*(flux-fw*u(i-1))/h;

end
% Visualisation


x=linspace(0,1,N);

figure
plot(x,sw_new,'LineWidth',2)
saveas(gcf, 'ma_figure_saturation_DDFVM_1D_injection.png');
xlabel('x')
ylabel('Saturation')
title('Transport saturation DDFVM')
grid on