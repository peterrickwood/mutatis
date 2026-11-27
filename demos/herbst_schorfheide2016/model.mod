// The small New Keynesian model of Herbst and Schorfheide (2016), as in model.yaml, in Dynare's language: the same equations,
// written for the Dynare import and for computing reference results with Dynare. Parameter values: the authors' posterior
// mode reported in their code's prior file (prior_dsge.txt); priors as in model.yaml.
var y pi R g z ygr infl int;
varexo eR eg ez;
parameters tau kappa psi1 psi2 rA piA gammaQ rho_R rho_g rho_z sigma_R sigma_g sigma_z;

tau = 2.83; kappa = 0.78; psi1 = 1.80; psi2 = 0.63; rA = 0.42; piA = 3.3; gammaQ = 0.52;
rho_R = 0.77; rho_g = 0.98; rho_z = 0.88; sigma_R = 0.22; sigma_g = 0.71; sigma_z = 0.31;

model(linear);
y = y(+1) - (1 / tau) * (R - pi(+1) - z(+1)) + g - g(+1);
pi = pi(+1) / (1 + rA / 400) + kappa * (y - g);
R = rho_R * R(-1) + (1 - rho_R) * (psi1 * pi + psi2 * (y - g)) + eR;
g = rho_g * g(-1) + eg;
z = rho_z * z(-1) + ez;
ygr = gammaQ + y - y(-1) + z;
infl = piA + 4 * pi;
int = piA + rA + 4 * gammaQ + 4 * R;
end;

shocks;
var eR; stderr sigma_R;
var eg; stderr sigma_g;
var ez; stderr sigma_z;
// measurement errors: a fifth of each series' sample standard deviation, as in the authors' code
var ygr; stderr 0.1159846;
var infl; stderr 0.2941664;
var int; stderr 0.4475874;
end;

varobs ygr infl int;

// The authors' priors. Dynare states the inverse gamma for a standard deviation by its mean and standard deviation; with
// nu = 4 those are 1.2533141373 s and 0.6551363776 s for Schorfheide's IG(s, nu).
estimated_params;
tau, gamma_pdf, 2, 0.5;
kappa, uniform_pdf, , , 0, 1;
psi1, gamma_pdf, 1.5, 0.25;
psi2, gamma_pdf, 0.5, 0.25;
rA, gamma_pdf, 0.5, 0.5;
piA, gamma_pdf, 7, 2;
gammaQ, normal_pdf, 0.4, 0.2;
rho_R, uniform_pdf, , , 0, 1;
rho_g, uniform_pdf, , , 0, 1;
rho_z, uniform_pdf, , , 0, 1;
sigma_R, inv_gamma_pdf, 0.5013256549262002, 0.2620545510248134;
sigma_g, inv_gamma_pdf, 1.2533141373155003, 0.6551363775620335;
sigma_z, inv_gamma_pdf, 0.6266570686577502, 0.32756818878101673;
end;

steady;
stoch_simul(order=1, irf=8, nograph, noprint);
