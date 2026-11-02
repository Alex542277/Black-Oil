clear all; clc; 


% Param?tres

Lx = 1; Ly = 1;
Nx = 50; Ny = 50;

dx = Lx/Nx;
dy = Ly/Ny;

phi = 0.2;
k = 300e-15;

mu_w = 1e-3;
mu_o = 5e-3;

T = 0.2;
dt = 0.0005;
Nt = floor(T/dt);

% Grille
[x,y] = meshgrid(linspace(dx/2,Lx-dx/2,Nx), ...
                 linspace(dy/2,Ly-dy/2,Ny));

% Initialisation

Sw = 0.2*ones(Ny,Nx);
Sw(:,1:3) = 1.0;   

P =-20*zeros(Ny,Nx);% Injection

%% Fonctions physiques
krw = @(S) S.^2;
kro = @(S) (1-S).^2;

% Boucle IMPES

for n = 1:Nt
    
    %% Mobilit?s
    lw = krw(Sw)/mu_w;
    lo = kro(Sw)/mu_o;
    lt = lw + lo;
    
    %% Pression implicite
    N = Nx*Ny;
    A = sparse(N,N);
    b = zeros(N,1);
    
    index = @(i,j) (j-1)*Nx + i;
    
    for j = 2:Ny-1
        for i = 2:Nx-1
            
            id = index(i,j);
            
            ltE = 0.5*(lt(j,i)+lt(j,i+1));
            ltW = 0.5*(lt(j,i)+lt(j,i-1));
            ltN = 0.5*(lt(j,i)+lt(j+1,i));
            ltS = 0.5*(lt(j,i)+lt(j-1,i));
            
            A(id,index(i+1,j)) = -k*ltE/dx^2;
            A(id,index(i-1,j)) = -k*ltW/dx^2;
            A(id,index(i,j+1)) = -k*ltN/dy^2;
            A(id,index(i,j-1)) = -k*ltS/dy^2;
            
            A(id,id) = k*(ltE+ltW)/dx^2 + k*(ltN+ltS)/dy^2;
        end
    end
    
    % Conditions Dirichlet pression
    for j=1:Ny
        idL = index(1,j);
        A(idL,idL)=1; b(idL)=1;
        
        idR = index(Nx,j);
        A(idR,idR)=1; b(idR)=0;
    end
    
    P_vec = A\b;
    P = reshape(P_vec,Nx,Ny)';
    
    %% Flux Darcy
    u = zeros(Ny,Nx+1);   % flux x
    v = zeros(Ny+1,Nx);   % flux y
    
    % Flux x
    for j=1:Ny
        for i=2:Nx
            lt_face = 0.5*(lt(j,i)+lt(j,i-1));
            dP = (P(j,i)-P(j,i-1))/dx;
            u(j,i) = -k*lt_face*dP;
        end
    end
    
    % Flux y
    for j=2:Ny
        for i=1:Nx
            lt_face = 0.5*(lt(j,i)+lt(j-1,i));
            dP = (P(j,i)-P(j-1,i))/dy;
            v(j,i) = -k*lt_face*dP;
        end
    end
    
    %% Transport saturation (Upwind 2D)
    Sw_new = Sw;
    
    for j=2:Ny-1
        for i=2:Nx-1
            
            % Fraction flow
            fw = lw(j,i)/lt(j,i);
            
            % Upwind X
            if u(j,i) > 0
                fwW = lw(j,i-1)/lt(j,i-1);
            else
                fwW = fw;
            end
            
            if u(j,i+1) > 0
                fwE = fw;
            else
                fwE = lw(j,i+1)/lt(j,i+1);
            end
            
            % Upwind Y
            if v(j,i) > 0
                fwS = lw(j-1,i)/lt(j-1,i);
            else
                fwS = fw;
            end
            
            if v(j+1,i) > 0
                fwN = fw;
            else
                fwN = lw(j+1,i)/lt(j+1,i);
            end
            
            flux = (fwE*u(j,i+1)-fwW*u(j,i))/dx + ...
                   (fwN*v(j+1,i)-fwS*v(j,i))/dy;
               
            Sw_new(j,i) = Sw(j,i) - dt/phi*flux;
        end
    end
    
    Sw_new = max(0,min(1,Sw_new));
    Sw = Sw_new;
    
end

figure
% view(45,30)     % angle 3D
% grid on
surf(x, y, Sw)
%saveas(gcf, 'ma_figure_3D_3_impes.png')
shading interp
colormap(jet)
colorbar

xlabel('x')
ylabel('y')
zlabel('Water Saturation')

title('Water Saturation - 3D Rendering')


view(45,35)
saveas(gcf, 'ma_figure_3D_4_impes.png')
camlight
lighting phong
material shiny

axis tight

if mod(n,50)==0
    
    surf(x,y,Sw)
    shading interp
    colormap(jet)
    zlim([0 1])
    
    title(['Time = ', num2str(n*dt)])
    xlabel('x'); ylabel('y'); zlabel('Sw')
    
    view(45,35)
    drawnow
    saveas(gcf, 'ma_figure_5D_impes.png')
    
end
