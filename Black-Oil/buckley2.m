%% VALIDATION CLASSIQUE : BUCKLEY-LEVERETT 1D + BILAN DE MASSE
% Script MATLAB/Octave autonome.
% Benchmark du transport scalaire classique de Buckley-Leverett.
% Ne valide pas, a lui seul, un solveur de pression 2D, DDFV, SPE ou une PDE fractionnaire.

clear; clc; close all;

%% 1. Parametres
L = 1.0;               % longueur du domaine (m)
Nx = 800;              % nombre de cellules
phi = 0.20;            % porosite
uT = 1.0;              % vitesse de Darcy totale (unites de test)
mu_w = 1e-3;           % viscosite eau (Pa.s)
mu_o = 5e-3;           % viscosite huile (Pa.s)
Swi = 0.20;            % saturation initiale
Sinj = 1.0;            % saturation imposee a l'entree
Tend = 0.10;           % temps final

if Nx < 2 || L <= 0 || phi <= 0 || uT <= 0 || Tend <= 0
    error('Verifier Nx, L, phi, uT et Tend : ces parametres doivent etre positifs.');
end
if Swi < 0 || Sinj > 1 || Swi >= Sinj
    error('Il faut 0 <= Swi < Sinj <= 1.');
end

dx = L/Nx;
x = ((1:Nx)' - 0.5)*dx;

%% 2. Lois constitutives
krw = @(S) S.^2;
kro = @(S) (1-S).^2;

% Forme algebraiquement stable de fw pour les mobilites choisies.
% fw = (S^2/mu_w) / (S^2/mu_w + (1-S)^2/mu_o)
fw = @(S) (mu_o*S.^2) ./ (mu_o*S.^2 + mu_w*(1-S).^2 + eps);

% Derivee analytique de fw, plus precise qu'une difference finie.
% fw'(S) = 2*mu_o*mu_w*S*(1-S) / denom^2
dfw = @(S) (2*mu_o*mu_w.*S.*(1-S)) ./ ...
    (mu_o*S.^2 + mu_w*(1-S).^2 + eps).^2;

%% 3. Saturation de choc : condition de tangence
% f'(S*) = (f(S*) - f(Swi))/(S* - Swi)
shock_fun = @(S) dfw(S) - (fw(S)-fw(Swi))./(S-Swi);

Sscan = linspace(Swi + 1e-7, Sinj - 1e-7, 10000);
gscan = arrayfun(shock_fun, Sscan);
idx = find(gscan(1:end-1).*gscan(2:end) <= 0, 1, 'first');

if isempty(idx)
    error('Impossible de localiser S*. Verifier les lois constitutives.');
end

Sshock = fzero(shock_fun, [Sscan(idx), Sscan(idx+1)]);
shock_speed = dfw(Sshock);

fprintf('Saturation de choc S* = %.8f\n', Sshock);
fprintf('Vitesse reduite du choc f''(S*) = %.8f\n', shock_speed);

%% 4. Solution numerique conservative : flux de Rusanov
Sw = Swi*ones(Nx,1);
mass_initial = phi*sum(Sw)*dx;

t = 0;
step = 0;
max_steps = 2e7;
CFL = 0.45;

% Vitesse maximale caracteristique calculee sur [0,1].
Stest = linspace(0,1,5001);
amax = max(abs(uT*arrayfun(dfw, Stest)));
if amax <= 0
    error('Vitesse caracteristique maximale invalide.');
end

% Integrales temporelles des flux numeriques de bord.
integrated_inflow = 0;
integrated_outflow = 0;

while t < Tend
    dt = CFL*phi*dx/amax;
    dt = min(dt, Tend-t);

    % Conditions aux limites :
    % gauche : saturation imposee Sinj ;
    % droite : sortie extrapolee (gradient nul).
    Sext = [Sinj; Sw; Sw(end)];
    SL = Sext(1:end-1);
    SR = Sext(2:end);

    FL = uT*fw(SL);
    FR = uT*fw(SR);
    Fhat = 0.5*(FL+FR) - 0.5*amax*(SR-SL);

    % Mise a jour finite-volume conservative.
    Sw_new = Sw - (dt/(phi*dx))*(Fhat(2:end)-Fhat(1:end-1));

    if any(~isfinite(Sw_new))
        error('Saturation non finie : verifier le pas de temps et le flux.');
    end
    if any(Sw_new < -1e-8) || any(Sw_new > 1+1e-8)
        error('Saturation hors [0,1]. Reduire CFL et verifier le schema.');
    end

    % Accumulation des flux de bord sur le pas de temps.
    integrated_inflow = integrated_inflow + dt*Fhat(1);
    integrated_outflow = integrated_outflow + dt*Fhat(end);

    Sw = min(1,max(0,Sw_new));
    t = t + dt;
    step = step + 1;

    if step > max_steps
        error('Nombre maximal de pas temporels depasse.');
    end
end

%% 5. Solution analytique entropique de Buckley-Leverett
% Variable de similarite xi = phi*x/(uT*t).
% Pour la rarefaction, f'(S)=xi. Le choc relie Sshock a Swi.
xi = phi*x/(uT*Tend);
S_exact = Swi*ones(Nx,1);
df_inj = dfw(Sinj);

for m = 1:Nx
    if xi(m) < df_inj
        % Etat d'injection derriere le bord de la rarefaction.
        S_exact(m) = Sinj;
    elseif xi(m) < shock_speed
        % Branche de rarefaction : inverser f'(S)=xi.
        % Pour ce modele, dfw est croissante sur la branche utile.
        fun = @(S) dfw(S)-xi(m);
        fa = fun(Sshock);
        fb = fun(Sinj);
        if fa*fb <= 0
            S_exact(m) = fzero(fun, [Sshock, Sinj]);
        else
            % Protection numerique si xi est hors de l'intervalle de la branche.
            S_exact(m) = Sshock;
        end
    else
        % Etat initial en avant du choc.
        S_exact(m) = Swi;
    end
end

xshock = (uT/phi)*shock_speed*Tend;
x_rarefaction_head = (uT/phi)*df_inj*Tend;

fprintf('\nPosition de la tete de rarefaction : %.8f m\n', x_rarefaction_head);
fprintf('Position analytique du choc         : %.8f m\n', xshock);
if xshock > L
    fprintf(['AVERTISSEMENT : le choc est hors du domaine [0,L] a Tend. ', ...
             'Le profil numerique ne contient donc pas le choc.\n']);
end

%% 6. Erreurs et bilan de masse
% Normes discretes sur le domaine, normalisees par L pour L1 et L2.
L1 = sum(abs(Sw-S_exact))*dx/L;
L2 = sqrt(sum((Sw-S_exact).^2)*dx/L);
Linf = max(abs(Sw-S_exact));

mass_final = phi*sum(Sw)*dx;
mass_change = mass_final - mass_initial;
mass_change_from_fluxes = integrated_inflow - integrated_outflow;
mass_residual = mass_change - mass_change_from_fluxes;

fprintf('\n--- Buckley-Leverett 1D ---\n');
fprintf('Cellules Nx                       : %d\n', Nx);
fprintf('Nombre de pas temporels           : %d\n', step);
fprintf('Temps final                       : %.10f\n', t);
fprintf('Erreur L1 normalisee              : %.6e\n', L1);
fprintf('Erreur L2 normalisee              : %.6e\n', L2);
fprintf('Erreur Linf                       : %.6e\n', Linf);
fprintf('Masse poreuse initiale            : %.10e\n', mass_initial);
fprintf('Masse poreuse finale              : %.10e\n', mass_final);
fprintf('Variation de masse                : %.10e\n', mass_change);
fprintf('Variation predite par les flux    : %.10e\n', mass_change_from_fluxes);
fprintf('Resid u du bilan de masse         : %.10e\n', mass_residual);

%% 7. Figures (sans xline, pour compatibilite MATLAB/Octave)
figure('Color','w');
plot(x, S_exact, 'k-', 'LineWidth', 2); hold on;
plot(x, Sw, 'r--', 'LineWidth', 1.5);
yl = [0 1];
if xshock >= 0 && xshock <= L
    plot([xshock xshock], yl, ':b', 'LineWidth', 1.5);
    legend('Analytic', 'Numerical solution', 'Position du choc', 'Location','best');
else
    legend('Analytic', 'Numerical solution', 'Location','best');
end
xlabel('x (m)');
ylabel('S_w');
title('Buckley-Leverett: Analytical and Numerical Solutions');
grid on;
ylim(yl);
saveas(gcf,'BL_analytique_vs_numerique.png');

figure('Color','w');
plot(x, abs(Sw-S_exact), 'LineWidth', 1.5);
xlabel('x (m)');
ylabel('|S_{w,num}-S_{w,exact}|');
title('One-time error - Buckley-Leverett');
grid on;
saveas(gcf,'BL_erreur_ponctuelle.png');

%% 8. Export des donnees (compatible avec versions sans table/writetable)
results_matrix = [x, Sw, S_exact, abs(Sw-S_exact)];
fid = fopen('BL_resultats.csv','w');
if fid < 0
    warning('Impossible de creer BL_resultats.csv');
else
    fprintf(fid,'x_m,Sw_numerique,Sw_analytique,erreur_absolue\n');
    fclose(fid);
    dlmwrite('BL_resultats.csv', results_matrix, '-append', ...
        'delimiter', ',', 'precision', '%.12g');
end

metrics_matrix = [Nx, phi, mu_w, mu_o, Tend, step, Sshock, shock_speed, ...
    L1, L2, Linf, mass_initial, mass_final, mass_change, ...
    integrated_inflow, integrated_outflow, mass_residual];
fid = fopen('BL_indicateurs.csv','w');
if fid < 0
    warning('Impossible de creer BL_indicateurs.csv');
else
    fprintf(fid,['Nx,phi,mu_w_Pa_s,mu_o_Pa_s,T_final,n_pas,S_choc,', ...
        'vitesse_choc_reduite,erreur_L1,erreur_L2,erreur_Linf,', ...
        'masse_initiale,masse_finale,variation_masse,flux_entree_integre,', ...
        'flux_sortie_integre,residu_bilan\n']);
    fclose(fid);
    dlmwrite('BL_indicateurs.csv', metrics_matrix, '-append', ...
        'delimiter', ',', 'precision', '%.12g');
end

fprintf('\nFichiers crees dans le dossier courant :\n');
fprintf('  BL_analytique_vs_numerique.png\n');
fprintf('  BL_erreur_ponctuelle.png\n');
fprintf('  BL_resultats.csv\n');
fprintf('  BL_indicateurs.csv\n');

fprintf(['\nINTERPRETATION : ce benchmark teste le transport scalaire 1D classique. ', ...
    'Il ne valide pas directement un solveur de pression 2D, la methode DDFV, ', ...
    'un cas SPE ou une PDE fractionnaire complete.\n']);
