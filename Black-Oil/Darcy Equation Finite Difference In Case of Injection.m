% R?solution de l'?quation de Darcy 1D par la m?thode des diff?rences finies
% -div(k grad(p)) = f

clear; clc;

% Param?tres physiques
L = 1.0;        % Longueur du domaine
k = 1.0;        % Perm?abilit? constante
f_val = -10.0; 

% Discr?tisation
Nx = 50; Ny = 50;
hx = L / (Nx - 1);
hy = L / (Ny - 1);

% Cr?ation des matrices
N = Nx * Ny;
A = sparse(N, N);
b = zeros(N, 1);

% Fonction index global
index = @(i,j) (j-1)*Nx + i;

for j = 1:Ny
    for i = 1:Nx
        idx = index(i,j);

        % Bord : Dirichlet p=0
        if i == 1 || i == Nx || j == 1 || j == Ny
            A(idx, idx) = 1;
            b(idx) = 0;
        else
            % Laplacien 5 points
            A(idx, index(i,j))   = -2*k/(hx^2) - 2*k/(hy^2);
            A(idx, index(i-1,j)) =  k/(hx^2);
            A(idx, index(i+1,j)) =  k/(hx^2);
            A(idx, index(i,j-1)) =  k/(hy^2);
            A(idx, index(i,j+1)) =  k/(hy^2);

            % Terme source
            b(idx) = -f_val;  % injection
        end
    end
end

% R?solution
p = A \ b;
P = reshape(p, [Nx, Ny]);

% Affichage
[X, Y] = meshgrid(0:hx:L, 0:hy:L);
figure;
surf(X, Y, P');
xlabel('x'); ylabel('y'); zlabel('Pression p');
title('Solution de l''equation de Darcy Injection');
shading interp; colorbar;
saveas(gcf, 'ma_figure_Darcy_production_DF.png'); 
