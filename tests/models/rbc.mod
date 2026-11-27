// A basic real business cycle model, written for mutatis's tests (the same model as rbc.yaml): log utility, fixed labour,
// Cobb-Douglas production, capital chosen this period and used next period, technology an AR(1) in logs.
var c k y i a;
varexo e;
parameters beta alpha delta rho sigma;
beta = 0.99; alpha = 0.33; delta = 0.025; rho = 0.95; sigma = 0.01;

model;
1 / c = beta * (1 / c(+1)) * (alpha * a(+1) * k ^ (alpha - 1) + 1 - delta);
y = a * k(-1) ^ alpha;
k = (1 - delta) * k(-1) + i;
i = y - c;
log(a) = rho * log(a(-1)) + e;
end;

steady_state_model;
a = 1;
k = (alpha / (1 / beta - 1 + delta)) ^ (1 / (1 - alpha));
y = k ^ alpha;
i = delta * k;
c = y - i;
end;

shocks;
var e; stderr sigma;
end;

steady;
stoch_simul(order=1, irf=0, nograph, noprint);
