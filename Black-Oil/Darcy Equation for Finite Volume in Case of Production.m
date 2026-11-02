% Implementation DDFVM pour l'équation de Darcy
clear all; clc;

%% Paramètres du problème
h = 0.05;  % taille de maille
N = 1/h;   % nombre de cellules par dimension

% Tenseur de perméabilité anisotrope
D = [1, 0; 0, 1];

%% Génération des grilles
% Grille primale (centres)
[xp, yp] = meshgrid(h/2:h:1-h/2, h/2:h:1-h/2);
% Grille duale (sommets)  
[xd, yd] = meshgrid(0:h:1, 0:h:1);

% Conversion en vecteurs
xp = xp(:); yp = yp(:);
xd = xd(:); yd = yd(:);

%% Dimensions
n_prim = length(xp);    % N^2 points primaux
n_dual = length(xd);    % (N+1)^2 points duaux  
total_dof = n_prim + n_dual;

%% Initialisation du système global
A_global = zeros(total_dof);
F_global = zeros(total_dof, 1);

%% Fonction source
f_source = @(x,y) 10;%% Production



%% ASSEMBLAGE DES EQUATIONS PRIMALES
for i = 1:N
    for j = 1:N
        % Index global du point primal courant
        idx_current = (j-1)*N + i;
        
        % Coefficients alpha (simplifi?s - tenseur constant)
        D_ij = D; D_ip1j = D; D_im1j = D; D_ijp1 = D; D_ijm1 = D;
        
        % Calcul des coefficients alpha selon formules pages 29-30
        alpha_center = 2*D_ij(2,2)*D_ijp1(2,2)/(D_ij(2,2)+D_ijp1(2,2)) + ...
                       2*D_ij(2,2)*D_ijm1(2,2)/(D_ij(2,2)+D_ijm1(2,2)) + ...
                       2*D_ij(1,1)*D_im1j(1,1)/(D_ij(1,1)+D_im1j(1,1)) + ...
                       2*D_ij(1,1)*D_ip1j(1,1)/(D_ij(1,1)+D_ip1j(1,1));
        
        alpha_north = -2*D_ij(2,2)*D_ijp1(2,2)/(D_ij(2,2)+D_ijp1(2,2));
        alpha_south = -2*D_ij(2,2)*D_ijm1(2,2)/(D_ij(2,2)+D_ijm1(2,2));
        alpha_east = -2*D_ij(1,1)*D_ip1j(1,1)/(D_ij(1,1)+D_ip1j(1,1));
        alpha_west = -2*D_ij(1,1)*D_im1j(1,1)/(D_ij(1,1)+D_im1j(1,1));
        
        alpha_ne = -(D_ij(2,2)*D_ijp1(2,1) + D_ijp1(2,2)*D_ij(2,1))/(D_ij(2,2)+D_ijp1(2,2)) - ...
                   (D_ij(1,1)*D_ip1j(1,2) + D_ip1j(1,1)*D_ij(1,2))/(D_ij(1,1)+D_ip1j(1,1));
        
        alpha_nw = (D_ij(2,2)*D_ijp1(2,1) + D_ijp1(2,2)*D_ij(2,1))/(D_ij(2,2)+D_ijp1(2,2)) + ...
                   (D_ij(1,1)*D_im1j(1,2) + D_im1j(1,1)*D_ij(1,2))/(D_ij(1,1)+D_im1j(1,1));
        
        alpha_se = (D_ij(2,2)*D_ijm1(2,1) + D_ijm1(2,2)*D_ij(2,1))/(D_ij(2,2)+D_ijm1(2,2)) + ...
                   (D_ij(1,1)*D_ip1j(1,2) + D_ip1j(1,1)*D_ij(1,2))/(D_ij(1,1)+D_ip1j(1,1));
        
        alpha_sw = -(D_ij(2,2)*D_ijm1(2,1) + D_ijm1(2,2)*D_ij(2,1))/(D_ij(2,2)+D_ijm1(2,2)) - ...
                   (D_ij(1,1)*D_im1j(1,2) + D_im1j(1,1)*D_ij(1,2))/(D_ij(1,1)+D_im1j(1,1));
        
        % Terme diagonal
        A_global(idx_current, idx_current) = alpha_center;
        
        % Voisins primaux NORD
        if j < N
            idx_north = (j)*N + i;
            A_global(idx_current, idx_north) = alpha_north;
        end
        
        % Voisins primaux SUD
        if j > 1
            idx_south = (j-2)*N + i;
            A_global(idx_current, idx_south) = alpha_south;
        end
        
        % Voisins primaux EST
        if i < N
            idx_east = (j-1)*N + (i+1);
            A_global(idx_current, idx_east) = alpha_east;
        end
        
        % Voisins primaux OUEST
        if i > 1
            idx_west = (j-1)*N + (i-1);
            A_global(idx_current, idx_west) = alpha_west;
        end
        
        % Voisins duaux NORD-EST
        if i < N && j < N
            idx_ne = n_prim + (j)*(N+1) + (i+1);
            A_global(idx_current, idx_ne) = alpha_ne;
        end
        
        % Voisins duaux NORD-OUEST
        if i > 1 && j < N
            idx_nw = n_prim + (j)*(N+1) + i;
            A_global(idx_current, idx_nw) = alpha_nw;
        end
        
        % Voisins duaux SUD-EST
        if i < N && j > 1
            idx_se = n_prim + (j-1)*(N+1) + (i+1);
            A_global(idx_current, idx_se) = alpha_se;
        end
        
        % Voisins duaux SUD-OUEST
        if i > 1 && j > 1
            idx_sw = n_prim + (j-1)*(N+1) + i;
            A_global(idx_current, idx_sw) = alpha_sw;
        end
        
        % Terme source
        x_coord = (i-0.5)*h;
        y_coord = (j-0.5)*h;
        F_global(idx_current) = h^2 * f_source(x_coord, y_coord);
    end
end

%% ASSEMBLAGE DES EQUATIONS DUALES
for i = 0:N
    for j = 0:N
        % Index global du point dual courant
        idx_current_dual = n_prim + j*(N+1) + i + 1;
        
        % ?quations duales seulement pour l'int?rieur
        if i > 0 && i < N && j > 0 && j < N
            % Coefficients beta (simplifi?s)
            D_ij = D; D_ip1j = D; D_im1j = D; D_ijp1 = D; D_ijm1 = D;
            D_ip1jp1 = D;
            
            % Coefficient diagonal beta (formule page 31)
            beta_center = (D_ijp1(2,2) + D_ip1jp1(2,2))/2 - ...
                         (D_ip1jp1(1,2) - D_ijp1(1,2))^2/(2*(D_ijp1(1,1) + D_ip1jp1(1,1))) + ...
                         (D_ij(2,2) + D_ip1j(2,2))/2 - ...
                         (D_ip1j(1,2) - D_ij(1,2))^2/(2*(D_ij(1,1) + D_ip1j(1,1))) + ...
                         (D_ip1jp1(1,1) + D_ip1j(1,1))/2 - ...
                         (D_ip1jp1(2,1) - D_ip1j(2,1))^2/(2*(D_ip1jp1(2,2) + D_ip1j(2,2))) + ...
                         (D_ij(1,1) + D_ijp1(1,1))/2 - ...
                         (D_ijp1(2,1) - D_ij(2,1))^2/(2*(D_ijp1(2,2) + D_ij(2,2)));
            
            A_global(idx_current_dual, idx_current_dual) = beta_center;
            
            % Voisins duaux NORD
            if j < N-1
                idx_dual_north = n_prim + (j+1)*(N+1) + i + 1;
                beta_north = -(D_ijp1(2,2) + D_ip1jp1(2,2))/2 + ...
                            (D_ip1jp1(1,2) - D_ijp1(1,2))^2/(2*(D_ijp1(1,1) + D_ip1jp1(1,1)));
                A_global(idx_current_dual, idx_dual_north) = beta_north;
            end
            
            % Voisins duaux SUD
            if j > 1
                idx_dual_south = n_prim + (j-1)*(N+1) + i + 1;
                beta_south = -(D_ij(2,2) + D_ip1j(2,2))/2 + ...
                            (D_ip1j(1,2) - D_ij(1,2))^2/(2*(D_ij(1,1) + D_ip1j(1,1)));
                A_global(idx_current_dual, idx_dual_south) = beta_south;
            end
            
            % Voisins duaux EST
            if i < N-1
                idx_dual_east = n_prim + j*(N+1) + i + 2;
                beta_east = -(D_ip1jp1(1,1) + D_ip1j(1,1))/2 + ...
                           (D_ip1jp1(2,1) - D_ip1j(2,1))^2/(2*(D_ip1jp1(2,2) + D_ip1j(2,2)));
                A_global(idx_current_dual, idx_dual_east) = beta_east;
            end
            
            % Voisins duaux OUEST
            if i > 1
                idx_dual_west = n_prim + j*(N+1) + i;
                beta_west = -(D_ij(1,1) + D_ijp1(1,1))/2 + ...
                           (D_ijp1(2,1) - D_ij(2,1))^2/(2*(D_ijp1(2,2) + D_ij(2,2)));
                A_global(idx_current_dual, idx_dual_west) = beta_west;
            end
            
            % Voisins primaux connect?s (formules pages 33-34)
            % Point (i,j)
            if i >= 1 && i <= N && j >= 1 && j <= N
                idx_prim_ij = (j-1)*N + i;
                beta_ij = -(D_ip1j(1,1)*D_ij(2,1) + D_ij(1,1)*D_ip1j(2,1))/(D_ij(1,1) + D_ip1j(1,1)) - ...
                          (D_ijp1(2,2)*D_ij(1,2) + D_ij(2,2)*D_ijp1(1,2))/(D_ij(2,2) + D_ijp1(2,2));
                A_global(idx_current_dual, idx_prim_ij) = beta_ij;
            end
            
            % Point (i,j+1)
            if i >= 1 && i <= N && j+1 >= 1 && j+1 <= N
                idx_prim_ijp1 = (j)*N + i;
                beta_ijp1 = (D_ijp1(1,1)*D_ip1jp1(2,1) + D_ip1jp1(1,1)*D_ijp1(2,1))/(D_ijp1(1,1) + D_ip1jp1(1,1)) + ...
                            (D_ijp1(2,2)*D_ij(1,2) + D_ij(2,2)*D_ijp1(1,2))/(D_ij(2,2) + D_ijp1(2,2));
                A_global(idx_current_dual, idx_prim_ijp1) = beta_ijp1;
            end
            
            % Point (i+1,j)
            if i+1 >= 1 && i+1 <= N && j >= 1 && j <= N
                idx_prim_ip1j = (j-1)*N + (i+1);
                beta_ip1j = (D_ip1j(1,1)*D_ij(2,1) + D_ij(1,1)*D_ip1j(2,1))/(D_ij(1,1) + D_ip1j(1,1)) + ...
                           (D_ip1j(2,2)*D_ip1jp1(1,2) + D_ip1jp1(2,2)*D_ip1j(2,1))/(D_ip1j(2,2) + D_ip1jp1(2,2));
                A_global(idx_current_dual, idx_prim_ip1j) = beta_ip1j;
            end
            
            % Point (i+1,j+1)
            if i+1 >= 1 && i+1 <= N && j+1 >= 1 && j+1 <= N
                idx_prim_ip1jp1 = (j)*N + (i+1);
                beta_ip1jp1 = -(D_ijp1(1,1)*D_ip1jp1(2,1) + D_ip1jp1(1,1)*D_ijp1(2,1))/(D_ijp1(1,1) + D_ip1jp1(1,1)) - ...
                              (D_ip1j(2,2)*D_ip1jp1(1,2) + D_ip1jp1(2,2)*D_ip1j(2,1))/(D_ip1j(2,2) + D_ip1jp1(2,2));
                A_global(idx_current_dual, idx_prim_ip1jp1) = beta_ip1jp1;
            end
            
            % Terme source dual
            x_coord_dual = i*h;
            y_coord_dual = j*h;
            F_global(idx_current_dual) = h^2 * f_source(x_coord_dual, y_coord_dual);
        end
    end
end

%% APPLICATION DES CONDITIONS AUX LIMITES DIRICHLET
% Conditions Dirichlet homogènes sur tous les bords

% Bords primaux (? = 0 sur ??)
for i = 1:N
    % Bord inf?rieur (j=1)
    idx_bottom = (1-1)*N + i;
    A_global(idx_bottom, :) = 0;
    A_global(idx_bottom, idx_bottom) = 1;
    F_global(idx_bottom) = 0;
    
    % Bord sup?rieur (j=N)
    idx_top = (N-1)*N + i;
    A_global(idx_top, :) = 0;
    A_global(idx_top, idx_top) = 1;
    F_global(idx_top) = 0;
end

for j = 1:N
    % Bord gauche (i=1)
    idx_left = (j-1)*N + 1;
    A_global(idx_left, :) = 0;
    A_global(idx_left, idx_left) = 1;
    F_global(idx_left) = 0;
    
    % Bord droit (i=N)
    idx_right = (j-1)*N + N;
    A_global(idx_right, :) = 0;
    A_global(idx_right, idx_right) = 1;
    F_global(idx_right) = 0;
end

% Bords duaux (? = 0 sur ??)
for i = 0:N
    % Bord inf?rieur dual (j=0)
    idx_dual_bottom = n_prim + 0*(N+1) + i + 1;
    A_global(idx_dual_bottom, :)=0;
    A_global(idx_dual_bottom, idx_dual_bottom) = 1;
    F_global(idx_dual_bottom) = 0;
    
    % Bord sup?rieur dual (j=N)
    idx_dual_top = n_prim + N*(N+1) + i + 1;
    A_global(idx_dual_top, :) = 0;
    A_global(idx_dual_top, idx_dual_top) = 1;
    F_global(idx_dual_top) = 0;
end

for j = 0:N
    % Bord gauche dual (i=0)
    idx_dual_left = n_prim + j*(N+1) + 0 + 1;
    A_global(idx_dual_left, :) = 0;
    A_global(idx_dual_left, idx_dual_left) = 1;
    F_global(idx_dual_left) = 0;
    
    % Bord droit dual (i=N)
    idx_dual_right = n_prim + j*(N+1) + N + 1;
    A_global(idx_dual_right, :) = 0;
    A_global(idx_dual_right, idx_dual_right) = 1;
    F_global(idx_dual_right) = 0;
end

%% RESOLUTION DU SYSTEME LINEAIRE
fprintf('R?solution du syst?me DDFV...\n');
fprintf('Taille du syst?me: %d x %d\n', total_dof, total_dof);
fprintf('Nombre de points primaux: %d\n', n_prim);
fprintf('Nombre de points duaux: %d\n', n_dual);

% Vérification que la matrice n'est pas singuli?re
if rank(A_global) < total_dof
    warning('La matrice est singulière ou mal conditionnée');
end

% R?solution
phi_solution = A_global \ F_global;

% Extraction des solutions primale et duale
phi_prim = phi_solution(1:n_prim);
phi_dual = phi_solution(n_prim+1:end);

% Reformater les coordonnées primales et duales en matrices
Xp = reshape(xp, N, N);
Yp = reshape(yp, N, N);
PhiP = reshape(phi_prim, N, N);

Xd = reshape(xd, N+1, N+1);
Yd = reshape(yd, N+1, N+1);
PhiD = reshape(phi_dual, N+1, N+1);

% Cr?er une grille fine pour interpolation
[Xplot, Yplot] = meshgrid(linspace(0,1,100), linspace(0,1,100));

% Interpoler la solution primale num?rique sur la grille fine
Ui_prim = interp2(Xp, Yp, PhiP, Xplot, Yplot, 'linear');

% Interpoler la solution duale num?rique sur la grille fine
Ui_dual = interp2(Xd, Yd, PhiD, Xplot, Yplot, 'linear');

% Moyenne pond?r?e des interpolations num?riques
Ui_final = (Ui_prim + Ui_dual)/2;


% Trac? solution num?rique finale
figure;
surf(Xplot, Yplot, Ui_final);
shading interp;
title('Solution numerique finale DDFV');
xlabel('x'); ylabel('y'); zlabel('\phi');
saveas(gcf, 'ma_figure_darcy_3D_DDFVM_production.png'); 
colorbar;
view(3);

