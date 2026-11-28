# Stage 4 gate: the base model's printed equations against the housing model's code

Written 2026-11-27, before building stage 4, as PLAN.md's stage 4 asks ("compare the base model's printed equations one by one
against the housing model's code, and count how many equations outside housing changed"). Kept as the evidence for the
decisions it leads to. Since then, decision 54 has handled most of Section 8's importer gaps (all but 2, 10, 12 and 13).
Scratch files mentioned below (the macro-expanded equations, preprocessor output, check runs) were not kept.

Base: Rees, Smith & Hall (2015), RDP 2015-07, Appendix A, equations A1 to A61 ("RSH").
Housing model: Gibbs, Hambur & Nodari (2018), RDP 2018-04 ("Reno"), public Dynare code, folder `Domestic Sector`.

## 0. Bottom line

1. **Many equations outside housing differ.** Of the 45 printed model equations (A1 to A45), 11 changed in substance for
   reasons that have nothing to do with housing, and 6 more changed in ways that have no effect at the published
   parameter values. Only 24 are identical. Taken literally, the gate's test ("if many did...") is met.
2. **But housing did not cause those changes.** None of the 11 substantive changes involves a housing variable. They
   are revisions of the base model: sticky wages, price indexation, a different foreign block, a different monetary
   policy rule, a modified interest parity condition, a persistent productivity-growth process, a different GDP ratio in
   the current account, and weights in the hours aggregate. The Reno paper says so itself (Section 4): to compare with
   RSH it used "a modified version of RSH that includes our sticky wage specification, inflation indexation, and the same
   foreign sector as we consider in this paper". The archive holds that modified three-sector model's Bayesian impulse
   responses (`bayes_irfs_three_sector.mat`), and they include sectoral wage observables, so the comparison model
   already had sticky wages.
3. **The housing step itself touches the rest of the model only through sums over sectors.** Housing variables enter 6
   printed non-housing equations (A8 hours aggregate, A21 resource market clearing, A23 aggregate investment, A24
   nominal GDP, A25 real GDP, A30 consumer price index), and every one of those entries is one more term in a sum over a
   list of sectors. In the Reno code those sums are literally macro loops over the sector lists. The household's
   consumption equations, the resource constraint, the policy rule, interest parity and the current account have no
   housing term.
4. **Verdict (moderate-to-high confidence): the block concept does not need rethinking on this evidence**, but the gate
   as worded mixes two steps. Base revisions (about 8 themes, each a replace-equation or replace-block edit) plus the
   housing step (extend three sector lists, replace one equation for the housing sector, untie one parameter, give
   housing its own investment shock, add two shock processes and four data links). Three gaps in mutatis's current edit
   vocabulary showed up (Section 5.4): `replace_block` insists on identical inputs and outputs, but the housing
   instances need extra inputs; steady-state calibration changes coefficient values throughout the model; and the
   n-sector data series are redefined when housing is split out of it.
5. **The housing model's published results can be matched exactly.** Dynare 6.5 (Octave), run on a scratch copy at the
   stored posterior mode, reproduces the stored Laplace approximation to within 1.4e-6 (-5139.2166551 against
   -5139.2166564). Exact matching needs three details: the stored estimation used **99 observations, 1992Q2 to
   2016Q4** (not the 100 the spreadsheet range and the paper suggest); each series is demeaned over those 99 quarters;
   and the steady-state values are computed once, at the calibrated habit b = 0.8, and then held fixed during
   estimation.

## 1. Sources and method

- RSH Appendix A was read from the PDF page images (pages 54 to 60 of the file, printed pages 46 to 52), not only from
  the text extraction. The main text (Sections 2 and 3, Tables 1 to 4, Appendix B) was used to judge misprints.
- The Reno code was macro-expanded by Dynare 6.5's preprocessor (`json=parse`) into 143 equations, numbered R001 to
  R143 in this report in the preprocessor's order.
- The Reno paper itself (RDP 2018-04, downloaded from rba.gov.au into `stage4/reno_paper/`) was used for the authors'
  statements about the base, the data definitions and Tables 1 to 5.

**Renaming used** (RSH symbol to code name): λ → lambda; habit h (and b) → b; ξc → xi_c; µ → mu; ŵ_j → w_j; aggregate
hours ĥ / Ĥ → h; Ĥ_j → h_j; λ^k_j → lambda_j; î_j → inve_j; Υ → upsilon; Φ → phik_n; α_j → alpha_h_j; γ_j →
alpha_k_j; (1−α_j−γ_j) → alpha_z_j; r̂_j (rental rate) → rk_j; r̂ (policy rate) → r; π̂ → infl; κ_j/100 → phi_pi_j
(= slope_pi_j/100); ε_πj → e_infl_j; ŷ^d_m → y_d_m; ŷ^x_m → y_x_m; p̂*_m → p_m_star; ŷ^x_z, ẑ_x → z_x; ŷ^n_z,
ŷ^m_z → z_n, z_m; î → inve; ŷ^nva → y_nva; ŷ^va → y_va; b* → b_star; r* → r_star (steady state rstar_bar); ν̂
(risk premium) → psi; ψ̂ (its shock) → psi_tilde; Δs → S_on_S_lag; ξ_y* → xi_y_star; e*_π → e_infl_star.
Timing: RSH's k̂_{j,t} (capital in place at t) is the code's k_j(-1) (Dynare's end-of-period stock). Shocks enter the
code as eps/100 with standard deviations in per cent; RSH reports standard deviations ×100. These are units, not
changes.

**Classes used:**
- **S, same**: identical up to renaming, the timing convention, shock units, or multiplying through by a constant.
- **M, misprint**: differs from the code, but the difference is an evident misprint in RSH's appendix (RSH's own text
  or neighbouring equations agree with the code).
- **H, housing terms only**: changed only by added terms in housing variables.
- **O, other change**: changed in a way that has nothing to do with housing. **Oi** marks changes with no effect at
  the published parameter values (a term multiplied by a parameter fixed at 0, a shock with zero variance, or an effect
  of order 1e-5).
- **R, removed**: no counterpart. (None; A2 is replaced, and counted under O.)

## 2. Equation-by-equation comparison

### 2.1 Summary table

| RSH | What it is | Reno code | Class | Note |
|---|---|---|---|---|
| A1 | marginal utility of consumption | R003 | S | RSH prints both h and b for the habit parameter (misprint) |
| A2 | labour supply, j∈{n,m,z} | R005–R007, R010–R012, R015–R017 | O | replaced by sticky wages: 3 equations per sector |
| A3 | investment supply, j∈{n,m,z} | R025–R027 | S | Φ → phik_n, shared by n, m, z; now calibrated (20), not estimated |
| A4 | production, j∈{n,m,z} | R008, R013, R018 | Oi | + aggregate technology `a` (switched off); land and a 1e-5 own-resource input in z |
| A5 | capital Euler (Tobin's q), j∈{n,m,z} | R029–R031 | S | common δ → sector δ_j (parameter values only) |
| A6 | capital accumulation, j∈{n,m,z} | R033–R035 | S | |
| A7 | bond Euler equation | R004 | M | sign of the right side misprinted in RSH; µ_{t+1} term see 2.2 |
| A8 | hours aggregate | R002 | H + O | + h_h term; + labour weights xi_l_j |
| A9 | Phillips curves n, m | R039, R043 | O | sector-specific indexation χ_j |
| A10 | marginal cost n, m | R040, R044 | Oi | − a (switched off) |
| A11 | labour demand n, m | R037, R041 | S | |
| A12 | resource-input demand n, m | R038, R042 | S | |
| A13 | import-price Phillips curve | R054 | O | indexation χ_f |
| A14 | import marginal cost | R055 | S | |
| A15 | hours in resources | R049 | S | |
| A16 | capital in resources | R050 | M | RSH prints r̂_z on the left; should be k̂_z (− µ̂) |
| A17 | domestic resource price | R051 | S | |
| A18 | foreign demand for manufactures | R063 | S | |
| A19 | domestic demand, j∈{n,m,f} | R064–R066 | S | housing instance R067 is new |
| A20 | manufactures market clearing | R068 | S | |
| A21 | resources market clearing | R069 | H + Oi | + z_h term; + z_z term (own input, 1e-5) |
| A22 | domestic final demand | R071 | S | |
| A23 | aggregate investment | R070 | H | + inve_h term |
| A24 | nominal value added | R072 | H + Oi | + housing term; y_z → y_va_z (identical here) |
| A25 | real value added | R073 | H + Oi | same as A24 |
| A26 | current account | R078 | O | ratios to real GDP y_va instead of nominal y_nva |
| A27 | interest parity | R079 | O | modified parity with phitilde (estimated, mode 0.069) |
| A28 | risk premium | R080 | S | |
| A29 | real exchange rate | R081 | S | |
| A30 | CPI identity | R058 | H | + p_h term (and a common factor 1−ζ) |
| A31 | relative prices, j∈{n,m,f} | R059–R061 | S | housing instance R062 is new |
| A32 | monetary policy rule | R082 | O | no output level; two-quarter inflation and growth |
| A33 | foreign IS curve | R084 + R086 | O | habit-based foreign household, two equations |
| A34 | foreign Phillips curve | R087 | O | indexation χ* |
| A35 | foreign policy rule | R089 | O | no output level; growth inside (1−ρ*) |
| A36 | productivity growth | R091 | O | white noise → AR(1), ρ_µ = 0.3332 |
| A37–A39 | sector technology a_n, a_m, a_z | R095, R097, R099 | S | housing instance R101 is new |
| A40 | consumption preference shock | R109 | S | |
| A41 | public demand | R083 | S | |
| A42 | risk-premium shock | R108 | S | |
| A43 | foreign resource price | R090 | Oi | + two terms with coefficients fixed at 0 |
| A44 | foreign demand shock | R085 | S | |
| A45 | foreign cost-push shock | R088 | S | |
| A46 | GDP growth | R112 | S | measurement error scale 0 |
| A47 | consumption growth | R113 | S | + measurement error (0.169) |
| A48 | investment growth | R114 | S | + error; investment now includes dwellings via A23 |
| A49 | non-tradeable value added | R131 | S | + error; **data now exclude housing** |
| A50 | non-resource tradeable value added | R133 | S | + error |
| A51 | resource value added | R135 | S | + error |
| A52 | non-resource exports | R122 | S | + error |
| A53 | resource exports | R121 | S | + error |
| A54 | inflation | R126 | S | error scale 0 |
| A55 | non-tradeable inflation | R128 | S | + error; **data now exclude housing** |
| A56 | cash rate | R130 | S | MA(1) error, scale 0 |
| A57 | nominal exchange rate | R124 | S | sign convention of the series (TWI up = appreciation); MA(1) error, scale 0 |
| A58 | resource prices | R140 | S | + error |
| A59 | foreign GDP | R141 | S | MA(1) error, scale 0 |
| A60 | foreign inflation | R142 | S | MA(1) error, scale 0 |
| A61 | foreign interest rate | R143 | S | + error (0.49) |

Measurement errors in A46–A61 are classed by the model part only. RSH's text says it put measurement errors on every
series except the two interest rates, but Appendix A prints none; Reno has errors on 20 of its 26 observed series
(zero on GDP, CPI inflation, the cash rate, the exchange rate, foreign GDP and foreign inflation; non-zero on the
foreign interest rate). In mutatis these are data links, not block equations.

### 2.2 Every equation that differs, with both forms

**A1 (S, with a misprint).** RSH: (µ−h)(µ−βb)λ̂_t = (µ−h)(µξ_c,t − hβξ_c,t+1) − (µ²+βh²)ĉ_t + µb(ĉ_t−1 + βĉ_t+1 −
µ̂_t + βµ̂_t+1). Code R003: `(mubar-b)*(mubar-beta*b)*lambda = (mubar-b)*(mubar*xi_c-beta*b*xi_c(1))-(mubar^2+beta*b^2)*c+mubar*b*(c(-1)+beta*c(1)-mu+beta*mu(1))`.
Identical once RSH's mixed h/b (both meaning the habit parameter; RSH eq. 1 calls it h) is read as one symbol.

**A2 (O, replaced).** RSH: ŵ_j,t = (η−σ)ĥ_t + σĤ_j,t − λ̂_t, j∈{n,m,z}. Code, for each sector (n shown, R005–R007):
`pi_w_n = ws_n + chi_w_n/(1+beta*chi_w_n)*pi_w_n(-1) + beta/(1+beta*chi_w_n)*pi_w_n(1) - phi_pi_w_n*(w_n - wg_n)`;
`pi_w_n = w_n - w_n(-1) + infl + mu`; `wg_n = -(lambda - xi_h - (eta-sigma)*h - sigma*h_n)`.
The wage gap wg_j is RSH's right-hand side plus a new labour-supply shock xi_h (the code keeps the flexible-wage line,
commented out: `w_j = xi_h + (eta - sigma)*h + sigma*h_j - lambda`). New with it: sectoral wage shocks ws_j (R096,
R098, R100), the labour-supply shock xi_h (R110), and four wage observables. Reno's paper attributes the wage
specification to Burgess et al. (2013).

**A4 (Oi).** RSH: ŷ_j,t = a_j,t + α_j H_j,t + γ_j k_j,t + (1−α_j−γ_j) z_j,t. Code n (R008):
`y_n = a + a_n + alpha_h_n*h_n + alpha_k_n*(k_n(-1)-mu) + alpha_z_n*z_n`; m likewise (R013); z (R018):
`y_z = a + a_z + alpha_h_z*h_z + alpha_k_z*(k_z(-1)-mu) + alpha_z_z*z_z + alpha_land_z*land_z`, with `land_z = 0`
(R052), `z_z = y_z` (R053), alpha_z_z = 0.00001.
Differences: (i) aggregate stationary technology `a`, inert because its shock has standard deviation 0 and ρ_a = 0;
(ii) the −µ on capital, which RSH's A11, A12 and A16 carry and A4 omits, so probably an RSH misprint (moderate
confidence); (iii) for z, RSH's printed A4 has a resource-input term, but RSH's own text (eq. 18) uses land with share
1−α_z−γ_z; the code follows the text (land fixed, share 0.5) and adds a resource input of size 1e-5.

**A7 (M).** RSH: r̂_t = E_t(λ̂_t+1 − π̂_t+1) − λ̂_t. Code R004: `-r = lambda(1) - infl(1) - lambda - mu(1)`, that is
r = λ_t − E λ_t+1 + E π_t+1 + E µ_t+1. RSH's printed sign is reversed (as printed, higher expected marginal utility
would raise the interest rate), so this is a misprint (high confidence). The µ_t+1 term has zero expectation under
RSH's white-noise µ (A36), so leaving it out is harmless in RSH. It matters in Reno only because A36 changed.

**A8 (H + O).** RSH: Ĥ_t = (H_n/H)^(1+σ) Ĥ_n,t + (H_m/H)^(1+σ) Ĥ_m,t + (H_z/H)^(1+σ) Ĥ_z,t. Code R002:
`h_ss^(1+sigma)*h = xi_l_n*h_n_ss^(1+sigma)*h_n + xi_l_m*h_m_ss^(1+sigma)*h_m + xi_l_z*h_z_ss^(1+sigma)*h_z + xi_l_h*h_h_ss^(1+sigma)*h_h`.
Housing adds the h_h term. Separately, the weights xi_l_j are new: they are solved in the steady state to hit
sectoral hours shares (with xi_l_j = 1 the code would equal RSH). Only the products Al·xi_l_j are determined (see
Section 6.6); the equation's coefficients do not depend on that free scale.

**A9 (O).** RSH: π̂_j,t = (κ_j/100) m̂c_j,t + β E_t{π̂_j,t} + ε_πj,t (the E_t{π̂_j,t} is a misprint for t+1). Code (n,
R039): `infl_n = phi_pi_n*mc_n + beta*infl_n(1)/(1+beta*chi_n) + chi_n/(1+beta*chi_n)*infl_n(-1) + e_infl_n`, with
`e_infl_n = rho_infl_n*e_infl_n(-1) + eps_e_n/100` and rho_infl_n = 0. The change is indexation. Note that RSH's
estimated model already had one common indexation parameter (Table 4: χ, posterior mean 0.24) that its Appendix A
leaves out; Reno has one χ per sector (n, m, h, f, exports). So against RSH's actual model the change is "common χ to
sector-specific χ_j".

**A10 (Oi).** RSH: m̂c_j = α_j ŵ_j + γ_j r̂_j + (1−α_j−γ_j) p_z − p̂_j − â_j. Code R040:
`mc_n = alpha_h_n*w_n + alpha_k_n*rk_n + alpha_z_n*p_z - p_n - a_n - a`. Only the inert −a.

**A13 (O).** RSH: π̂_f,t = (κ_f/100) m̂c_f,t + β E π̂_f,t+1 + ε_πf,t. Code R054:
`infl_f = phi_pi_f*mc_f + beta*infl_f(1)/(1+beta*chi_f) + chi_f/(1+beta*chi_f)*infl_f(-1) + e_infl_f`. Indexation.

**A16 (M).** RSH prints: r̂_z,t = p̂_z,t + ŷ_z,t − r̂_z,t. Code R050: `k_z(-1) - mu = y_z + p_z - rk_z`. The left side
must be capital (RSH's heading says "Capital used in the resources sector"), with −µ̂_t as in A11. Misprint (high
confidence for the variable, moderate for the −µ).

**A21 (H + Oi).** RSH: y_z ŷ_z = y^x_z ŷ^x_z + y^m_z ŷ^m_z + y^n_z ŷ^n_z. Code R069:
`y_z*y_z_ss = z_z*z_z_ss + z_x_ss*z_x + z_n*z_n_ss + z_m*z_m_ss + z_h*z_h_ss`. Housing adds z_h (the housing
sector uses resources). The z_z term is the 1e-5 own-input of the resource sector.

**A23 (H).** RSH: î = (I_n/I) î_n + (I_m/I) î_m + (I_z/I) î_z. Code R070:
`inve_ss*inve = inve_n*inve_n_ss + inve_m*inve_m_ss + inve_z*inve_z_ss + inve_h*inve_h_ss`.

**A24 (H + Oi).** RSH: ŷ^nva = (p_n y^nva_n/y^nva)(p̂_n + ŷ^nva_n) + (p_m y^nva_m/y^nva)(p̂_m + ŷ^nva_m) +
(p_z y_z/y^nva)(p̂_z + ŷ_z). Code R072: `y_nva_ss*y_nva = Σ_{j∈n,m,z,h} p_j_ss*y_va_j_ss*(p_j + y_va_j)`. Housing adds a
term. The resource term uses y_va_z rather than y_z, but R019 and R053 make y_va_z equal to y_z exactly.

**A25 (H + Oi).** RSH: ŷ^va = (p_n y^nva_n/y^nva) ŷ^va_n + (p_m y^va_m/y^nva) ŷ^va_m + (p_z y_z/y^nva) ŷ_z. Code R073:
`y_va_ss*y_va = Σ_{j∈n,m,z,h} p_j_ss*y_va_j_ss*y_va_j`, with y_va_ss = y_nva_ss. As A24.

**A26 (O).** RSH: b*_t+1/r* = b*_t/(π*µ) + (p_z z_x/y^nva)(p̂_z + ẑ_x − ŷ^nva) + (p*_m y^x_m/y^nva)(p̂*_m + ŷ^x_m −
ŷ^nva) − (q y_f/y^nva)(q̂ + ŷ_f − ŷ^nva). Code R078:
`b_star/(psi_ss*rstar_bar) = b_star(-1)/(mubar*pibar_star) + p_z_ss*z_x_ss/y_nva_ss*(p_z+z_x-y_va) + p_m_ss*y_x_m_ss/y_nva_ss*(p_m_star+y_x_m-y_va) - q_ss*y_f_ss/y_nva_ss*(q+y_f-y_va)`.
The deviations are taken relative to real GDP (y_va) instead of nominal GDP (ŷ^nva). RSH's text defines b* relative
to nominal GDP (P_t Y_t), so RSH's printed form is the consistent one; the code's is a simplification or a slip (cannot
tell which). psi_ss = 1, so the left sides agree.

**A27 (O).** RSH: E q̂_t+1 − q̂_t = E(π̂*_t+1 − π̂_t+1) + r̂ − r̂* − ν̂ (with A29, this is r̂ − r̂* − ν̂ = E Δs_t+1).
Code R079: `r - r_star - psi = (1-phitilde)*S_on_S_lag(1) - phitilde*S_on_S_lag`. Equal to RSH when phitilde = 0;
phitilde is estimated (prior beta(0.5, 0.15), mode 0.069). Reno's paper prints this modified parity as its eq. 27.

**A30 (H).** RSH: 0 = Σ_{j∈n,m,f} ω_j p_j^(1−ζ) p̂_j. Code R058: `0 = Σ_{j∈n,m,f,h} p_j*(1-zeta)*omega_j*p_j_ss^(1-zeta)`.

**A32 (O).** RSH: r̂ = ρ_r r̂_t−1 + (1−ρ_r)(φ_π π̂ + φ_y ŷ^va) + φ_Δy(ŷ^va − ŷ^va_t−1) + φ_q(q̂ − q̂_t−1) + ε_r. Code R082:
`r = rho_r*r(-1) + (1-rho_r)*(rho_pi/2*(infl+infl(-1)) + rho_deltay/2*(y_va-y_va(-2)) + rho_q*(q-q(-1))) + eps_r/100`.
Four differences: no output-level term (rho_y is declared and set to 0.12 but unused); inflation averaged over two
quarters; output growth over two quarters (halved); growth and exchange-rate terms inside (1−ρ_r). Reno's own paper
(eq. 20) prints a quarterly rule without the two-quarter averaging, so Reno's paper and code also disagree here.

**A33 (O).** RSH: ŷ* = E ŷ*_t+1 − (r̂* − E π̂*_t+1) + ξ*_y − E ξ*_y,t+1. Code, two equations, R084:
`(mubar-hab_star)*(mubar-beta*hab_star)*lambda_star = (mubar-hab_star)*(mubar*xi_y_star-beta*hab_star*xi_y_star(1)) - y_star*(mubar^2+beta*hab_star^2) + mubar*hab_star*(beta*mu(1)+y_star(-1)+beta*y_star(1)-mu)`
and R086: `-r_star = lambda_star(1) - infl_star(1) - lambda_star - mu(1)`. With hab_star = 0 these reduce to RSH's A33
plus a µ_t+1 term. hab_star = 0.8043 is fixed ("taken from other estimation").

**A34 (O).** RSH: π̂* = β E π̂*_t+1 + (κ*/100) ŷ* + e*_π. Code R087:
`infl_star = beta*infl_star(1)/(1+beta*chi_star) + chi_star/(1+beta*chi_star)*infl_star(-1) + y_star*phi_pi_star/(1+beta*chi_star) + e_infl_star`,
chi_star = 0.2725. Indexation.

**A35 (O).** RSH: r̂* = ρ*_r r̂*_t−1 + (1−ρ*_r)(φ*_π π̂* + φ*_y ŷ*) + φ*_Δ(ŷ* − ŷ*_t−1) + ε*_r. Code R089:
`r_star = rho_r_star*r_star(-1) + (1-rho_r_star)*(infl_star*rho_pi_star + rho_deltay_star*(y_star-y_star(-1))) + eps_r_star/100`.
No output-level term (rho_y_star declared, set to 0, unused); growth term inside (1−ρ*).

**A36 (O).** RSH: µ_t = ε_µ,t. Code R091: `mu = rho_mu*mu(-1) + eps_mu/100`, rho_mu = 0.3332.

**A43 (Oi).** RSH: p*_z = ρ p*_z,t−1 + φ_zy ε_y* + ε_p*. Code R090:
`p_star_z = rho_p_star_z*p_star_z(-1) + eps_p_star_z/100 + rho_z_y*eps_y_star/100 + rho_z_pi*eps_infl_star/100 + rho_z_r*eps_r_star/100`,
with rho_z_pi = rho_z_r = 0.

**A57 (S).** RSH: Δs^obs = 100(s_t − s_t−1). Code R124: `ntwi_growth = -100*S_on_S_lag + sd_mentwi/sqrt(2)*(mentwi-mentwi(-1))`,
sd_mentwi = 0. The sign flip matches the data (the trade-weighted index rises when the currency appreciates).

**Data definitions behind unchanged equations (A49, A55).** RSH's non-tradeable value added includes construction, real
estate and ownership of dwellings (RSH Appendix B). Reno's "non-tradeable excluding housing" series removes them
(Reno Appendix A). So the equations are unchanged but the series they observe changed, because of housing. RSH's
foreign series were also redefined (PPP-weighted major trading partners and G7 inflation in RSH; export-weighted major
trading partners in Reno), which has nothing to do with housing.

### 2.3 Reno equations with no counterpart in A1 to A61

All 143 Reno equations are accounted for: 70 correspond to A1–A45 (counting A2's nine replacement equations and A33's
two), 16 to A46–A61, and 57 are listed here.

**Housing (23).** Of these, 16 are the generic sector equations instantiated for h, 3 are new in form, and 4 are
observation equations.
- Instances of the sector templates (16): wage Phillips curve, wage inflation, wage gap (R020–R022); production
  (R023); value added (R024); investment supply (R028, **with its own adjustment cost phik_h and its own investment
  shock upsilon_h**); capital accumulation (R036, **with upsilon_h**); labour demand (R045); resource-input demand
  (R046); Phillips curve (R047); marginal cost (R048); relative price (R062); demand for housing services (R067,
  `y_h = -zeta*p_h + dfd`); technology, wage and cost-push shock processes (R101, R102, R107).
- New in form (3): the housing capital Euler equation with housing in utility (R032, below), the housing investment
  shock process `upsilon_h` (R094) and the housing preference shock process `xi_house` (R111).
- Observation (4): `inve_h_growth` (R115), `infl_h_obs` (R127), `y_va_h_growth` (R137), `pi_w_h_obs` (R138).

R032, in the code's own form:
```
lambda_h = (k_h_ss*lambda_ss*(-mubar+bH)^2*(mubar*(-mu(+1)+rk_h(+1))+lambda(+1)*(mubar+beta*(-1+delta_h))
            +(-lambda_h(+1)+rk_h(+1))*beta*(-1+delta_h))-(mubar)^2*(k_h*mubar-mu(+1)*mubar
            +mubar*rk_h(+1)+lambda(+1)*(mubar-bH)-k_h(-1)*bH+mu*bH+mu(+1)*bH-rk_h(+1)*bH+xi_house(+1)*(-mubar+bH))*beta*Ah)
           /(k_h_ss*lambda_ss*mubar*(-mubar+bH)^2);
```
It uses, beyond the A5 template's variables (lambda, rk_h, lambda_h, mu): the housing stock **k_h and k_h(-1)**, the
preference shock **xi_house(+1)**, the new parameters Ah (fixed 0.3) and bH (estimated), and the steady-state values
k_h_ss and lambda_ss. With Ah = 0 it reduces to the A5 form.

**In RSH's model, but missing from RSH's Appendix A (6).** RSH's text and tables imply these, and the appendix uses
their variables without printing their equations: value added of n and m (R009, R014; RSH text: Y^va_m = Y_m −
(P_z/P_m) Z_m); the export-price Phillips curve and the relative export price (R056, R057; RSH says export prices are
sticky in local currency and estimates κ_πm* and σ_πm* in Table 4, while A18 and A26 use p̂*_m with no equation for
it); the investment-shock process Υ (R093; RSH eq. 40 lists it, A36–A45 omit it); and the public-demand observation
(R116; RSH Appendix B lists public demand among the observed series, A46–A61 omit it). R056 has its own indexation
parameter χ_m*, where RSH used one common χ; the rest match RSH's text. So **RSH's Appendix A is not a complete statement of RSH's estimated model**:
at least 5 model equations and 1 observation equation are missing, and it leaves out the indexation and the
measurement errors that its Table 4 and text report.

**Other non-housing additions (4):** wage shock processes ws_n, ws_m, ws_z (R096, R098, R100) and the labour-supply
shock xi_h (R110), all part of the sticky-wage change.

**Inert additions (8):** aggregate technology `a` (R092, shock switched off); `land_z = 0` (R052); `z_z = y_z` (R053);
value added of z (R019, identical to y_z); AR(1) cost-push processes with ρ fixed at 0 for f, exports, n, m (R103–R106;
equivalent to RSH's white-noise mark-up shocks).

**Hours and extra observation equations (6):** measured total hours (R001, a sum over production sectors including
h_h); observed hours growth (R139), tradeable inflation (R129), sectoral wage growth n, m, z (R132, R134, R136).

**Unobserved definitions (10):** export and import values and volumes (R074–R077) and their growth rates (R117–R120),
total_hours_obs (R123), q_obs (R125). None is in `varobs`.

## 3. Counts

| | Printed equations | Sector instances |
|---|---|---|
| A1–A45 model equations | 45 | 63 |
| identical (S) | 24 | 36 |
| differ only by a misprint in RSH (M) | 2 (A7, A16) | 2 |
| housing terms only (H) | 2 (A23, A30) | 2 |
| housing terms + inert change (H + Oi) | 3 (A21, A24, A25) | 3 |
| housing terms + substantive non-housing change (H + O) | 1 (A8) | 1 |
| substantive non-housing change (O) | 10 (A2, A9, A13, A26, A27, A32, A33, A34, A35, A36) | 13 |
| inert non-housing change only (Oi) | 3 (A4, A10, A43) | 6 |
| removed | 0 | 0 |
| A46–A61 observation equations | 16 | |
| same model part (S) | 16 | |
| of which observed series redefined because of housing | 2 (A49, A55) | |

- **Changed outside housing, substantively: 11 printed equations** (A2, A8, A9, A13, A26, A27, A32, A33, A34, A35,
  A36; 14 sector instances). None contains a housing variable.
- **Changed outside housing, inertly: 6** (A4, A10, A43, and the inert parts of A21, A24, A25).
- **Non-housing equations that gain housing terms: 6** (A8, A21, A23, A24, A25, A30), all through sums over a sector
  list.
- **New in the Reno code: 57 equations**: 23 housing (16 template instances, 3 new in form, 4 observation), 6 that
  belong to RSH's model but are missing from its appendix, 4 non-housing additions, 8 inert, 6 hours and extra
  observation equations, 10 unobserved definitions.

## 4. What the evidence says about the cause of the non-housing changes

- The Reno paper, Section 4: "we use a modified version of RSH that includes our sticky wage specification, inflation
  indexation, and the same foreign sector as we consider in this paper." That covers A2, A9, A13, A33–A35 directly.
- The three-sector comparison model's Bayesian responses (`Bayesian IRFs/bayes_irfs_three_sector.mat`, dated June 2017)
  include `pi_w_n_obs`, `pi_w_m_obs`, `pi_w_z_obs`, `hours_growth`, `infl_m_obs`. So that model had sticky wages, the
  hours observable and tradeable inflation, but no housing. Its solution has 121 rows (state and policy) and 49 shocks;
  the housing model's has 150 and 58. The difference of 9 shocks equals the 9 housing shocks and measurement errors
  (eps_xi_house, eps_upsilon_h, eps_a_h, eps_ws_h, eps_e_h, meinveh, meinflh, meyva_h, mepiwh). That fits only by
  count, since the file stores no names. The difference of 29 rows is 23 housing variables plus 6 that might be the
  six lagged-measurement-error auxiliaries. That last part is a guess (low confidence).
- The remaining non-housing changes (A26 ratio, A27 parity, A32 policy rule, A36 growth process, A8 weights, inert
  terms) are not named in the paper. None involves a housing variable, and the code writes them as edits of the base
  rather than as housing additions. I cannot rule out that some were made while building the housing model; the code
  history is not public.
- The code comments point to an RBA-internal lineage: "Adapted from code written by Dan Rees"; the Forecasting copy of
  the model file says its equation numbers refer to "'The Stationary System', pp.21 and following in the current
  notes", not to RSH's Appendix A.

Confidence that the 11 substantive non-housing changes were not caused by housing: **moderate to high** (direct for 6
of them from the paper's statement; by content for the other 5).

## 5. Assessment

### 5.1 Can base to housing be written as edits on blocks joined through ports?

**Housing step proper (modified three-sector base to Reno): yes, with three gaps in the current edit vocabulary.**
Confidence moderate to high. Directly verified: the only places housing variables enter non-housing equations are sums
over sector lists. Not tested: the edits themselves in mutatis.

The housing step as edits, against the code's own base:
1. `extend_set` on the three sector lists: production [n, m, z] → [n, m, z, h]; imperfectly competitive [n, m] →
   [n, m, h]; consumed [n, m, f] → [n, m, f, h]. That creates the 16 template instances and the extra terms in the 6
   sums (A8, A21, A23, A24, A25, A30) and in measured hours (R001).
2. Replace the h instance of the capital-Euler template (A5) by R032 (housing in utility). New parameters Ah (fixed)
   and bH (estimated).
3. `untie` the investment adjustment cost for h: phik_h, estimated (normal(3, 1.5)), where n, m, z share phik_n = 20.
4. Give the h instances of investment supply (A3) and capital accumulation (A6) the shock upsilon_h instead of the
   shared upsilon.
5. Add two shock processes (xi_house, upsilon_h) with their priors.
6. Add four data links (dwelling investment, housing value added, housing CPI inflation, housing wages) with their
   measurement errors.
7. Replace the data behind the n-sector value-added and inflation links with the "excluding housing" series.
8. Recalibrate the steady state: new targets (housing share of value added and of hours, housing share of capital via
   Ah), and the n-sector targets split (`non_tradeable_share = 0.64*(1-0.28)`, `housing_share = 0.64*0.28` in
   calcss_external.m).

**From RSH as printed to Reno: not a short list.** About eight themed replacements come first, each expressible as
existing edits:

| Theme | Equations | Edit type |
|---|---|---|
| sticky wages | A2 ×3 → 9 equations, + 4 shocks, + 3 observables | replace block (new outputs) |
| price indexation | A9 ×2, A13, A34, export Phillips curve | replace equation in the Phillips-curve template |
| foreign block | A33 (→ 2 equations), A34, A35 | replace block |
| monetary policy rule | A32 | replace equation |
| interest parity | A27 | replace equation |
| productivity growth process | A36 | replace equation |
| current-account ratio | A26 | replace equation |
| hours-aggregate weights | A8 | replace equation, + calibrated weights |
| inert extras | A4, A10, A43, z-sector land and own input | may be left out |

There are also parameter-level changes: sector-specific depreciation; a calibrated φ = 20 for n, m, z (RSH estimated
Φ = 1.78); and measurement errors on 20 of 26 series.

### 5.2 Which existing blocks need equation replacements for housing, and why

- **None, if aggregates are written as sums over sector lists.** The hours bundle (A8), resource market clearing
  (A21), aggregate investment (A23), nominal and real value added (A24, A25) and the CPI identity (A30) each gain a
  housing term only because housing is one more sector in the list they sum over. If these are instead written term by
  term, as in RSH's printed appendix, each of the six needs a `replace_equation`.
- **Household block:** no replacement. Housing utility is additively separable, so A1 (marginal utility of
  consumption) and A7 (bond Euler) are unchanged. The budget constraint does not appear in the log-linear system.
- **Resource constraint / domestic final demand (A22):** unchanged. Dwelling investment enters through aggregate
  investment (A23, a sum over sectors).
- **Taylor rule, interest parity, current account:** unchanged. Housing is non-traded.
- **Housing-sector instances:** the capital Euler (new form), investment supply (own φ and shock) and capital
  accumulation (own shock) differ from the template.

### 5.3 Strongest case against this verdict

The paper and code show that, in practice, an expansion arrived bundled with about eight unrelated base revisions, and
the intermediate model (RSH plus revisions, no housing) is not public. So stage 4's acceptance condition ("both
published models are reproduced and the step from one to the other is a list of edits") can only be met as: RSH (from
the paper, approximate) → base-revision edits → intermediate (checkable only approximately, against the three-sector
Bayesian IRFs) → housing edits → Reno (checkable exactly). The edit list is real but long, and its middle is unverified.
This does not show that ports are insufficient. It shows that the gate's single count conflates two steps.

### 5.4 Gaps this exposes in mutatis (from reading `mutatis/spec/edits.py`; not tested)

1. **`replace_block` requires identical inputs and outputs** (`_same_ports`). The housing Euler needs k_h and xi_house
   as extra inputs, and the h instances of investment supply and capital accumulation take upsilon_h instead of
   upsilon. So the instance replacement in `docs/design/specification-sketches.md` (Section 7, "same inputs and
   outputs") would be rejected as written. Options: allow an instance override to add inputs; an "untie" for shared
   input variables, not only parameters; or write the base templates with sector-indexed shocks bound together.
2. **Steady-state calibration changes coefficients everywhere.** The log-linear coefficients (c_ss/dfd_ss,
   inve_j_ss/inve_ss, p_j_ss·y_va_j_ss/y_nva_ss, ...) come from a 38-equation nonlinear steady-state solve with
   calibration targets (Section 6.6). Adding housing moves all of them (ω_n goes from 0.63 in RSH to 0.397). If
   steady-state values are plain numbers in the spec, the housing expansion must also overwrite dozens of numbers in
   non-housing equations. If they are computed from targets, mutatis needs calibration targets as part of the spec.
   This, not the equation count, is the larger threat to "an expansion is a short list of edits".
3. **Splitting a sector redefines existing data links.** The n-sector observables switch to "excluding housing"
   series, so the expansion includes data edits on existing links.
4. The sketch's `add_data: housing_prices` does not match Reno: the model has no house-price observable. Its four
   housing observables are dwelling investment, housing value added, housing CPI inflation and housing wages.

## 6. The housing model's Dynare code (inventory)

### 6.1 Entry points and macro processor
- **Entry point for everything:** `multisector.mod`. It is switched by two macro flags in the file:
  `@#define estimate_mod = 0` (line 13) and `@#define irfs = 0` (line 16). The read-me calls the second one
  `define_irfs`.
  - estimate_mod = 1 → `estimate_model.mod`: full estimation.
  - estimate_mod = 0, irfs = 0 → `run_model.mod`: `mode_file=multisector_mode, mode_compute=0, mh_replic=0,
    load_mh_file`, which reloads the stored mode and MH draws.
  - estimate_mod = 0, irfs = 1 → `run_model_bayes_irfs.mod`: as run_model, plus `bayesian_irf,
    irf_shocks=(eps_r eps_xi_house eps_a_h eps_upsilon_h eps_upsilon eps_p_star_z eps_psi)`.
  - Always included at the end: `check_calibration_targets_loglinear.mod`, about 200 lines of native MATLAB checks
    (fprintf, eval, if) wrapping two `shocks` blocks and two `stoch_simul` calls.
- **Foreign stage (separate model):** `Foreign Sector/multisector_foreign.mod`: 11 endogenous variables, 9 shocks, 4
  observed series, 15 estimated parameters; its posterior means are typed into
  `dynamic_calibrations_forestimation.mod` and the domestic `shocks` block. **This foreign-stage model is not the
  foreign block used in the domestic model.** The stage-1 IS curve has no habit, while the domestic file's has habit
  hab_star = 0.8043, which is not estimated in stage 1. The stage-1 foreign policy rule uses two-quarter averages,
  while the domestic file's is quarterly. So re-running stage 1 and feeding it to stage 2 is not one consistent model.
- **Macro features used:** `@#include` (6 files plus 1 of 3 conditional), `@#define` (lists of strings),
  `@#for j in list`, `@#if / @#else / @#endif`, and string comparison inside a loop (`@#if j == "z"`).
- **Model-local variables (`#`):** 12 in the source, 11 in the JSON (unused `rbar` dropped): rstar_bar,
  phi_pi_{n,m,f,h} = slope_pi_j/100, phi_pi_w_{n,m,z,h} = slope_pi_w_j/100, phi_pi_star = kappa_star/100,
  phi_pi_m_star = slope_pi_m_star/100.
- `model(linear)`: all steady states are 0.

### 6.2 Leads, lags, auxiliary variables
- Leads: one period only, on endogenous variables (including shock processes such as xi_c(+1), xi_house(+1), mu(+1)).
- Lags beyond one: `y_va(-2)` in the policy rule.
- Lagged exogenous variables: six measurement errors in MA(1) form, `sd/sqrt(2)*(me - me(-1))`, for mehours, mentwi,
  meq, mer, meystar, meinflstar. All six have scale 0 at the published calibration.
- **aux_vars** (`json=transform`, Dynare 6.5, needs `nostrict`): 7. One endogenous lag (type 1, orig `y_va(-1)`,
  giving y_va(-2)) and six exogenous lags (type 3: mehours, mentwi, meq, mer, meystar, meinflstar). This matches
  `M_.aux_vars` in the stored results exactly. **With `json=parse` (what mutatis uses) `aux_vars` is empty**: they are
  created at the transform step.
- Dynare 6.5 rejects the file at the transform step: "eps_z_x not used in model block". It needs `nostrict`.
  Dynare 4.4, which the authors used, did not check.

### 6.3 Steady state and calibrated parameters
- No `steady_state_model`. `initval` sets every endogenous variable and the measurement-error shocks to 0 (`a` is
  listed twice); then `steady(solve_algo = 3); check;`.
- **Steady-state values come from MATLAB code:** `[ssvals,~] = Manualsteadystate;` (a native statement) calls
  `LMFsolve('calcss_external', ...)`. That solves 38 equations in 38 unknowns: steady-state quantities and prices, plus
  the calibrated parameters ω_n, ω_m, ω_h, gbar, ω_m*, Al, xi_l_{n,m,z,h}. The targets are hard-coded in
  `calcss_external.m` lines 83–95. The solution is cached in the global `M_.old_steady_state` between calls.
  - 31 parameters are set as `name = ssvals(i)`; 3 more (`k_n_ss`, `k_m_ss`, `k_z_ss`) are not declared parameters, so
    they become native MATLAB assignments; 4 entries (wages) are unused.
  - 28 more parameters are expressions of those (c_ss, inve_j_ss, y_j_ss, ...). Also computed: beta =
    mubar·pibar/1.06^(1/4), the alpha_z_j shares, ω_f = 1 − Σω.
  - In all: 203 parameter assignments, 144 numeric, 31 from ssvals, 28 expressions; 14 parameters assigned twice.
- **The steady state is computed once, at the calibrated values (b = 0.8), and frozen.** Estimation does not recompute
  it, although b is estimated (0.90 at the mode) and enters the steady state through consumption. Exact reproduction
  must freeze it too.
- **The steady-state system is singular in one direction.** With η = σ = 1, only the products Al·xi_l_j are
  determined. Dynare 6.5 found Al = 16.54 where the stored run (and Reno Table 1, "A_H = 4.4177") has 4.4177. The
  products agree to 2e-8 and every model coefficient is unaffected. The other steady-state values agree with the
  stored ones to 5e-8 (solver tolerance 1e-6), and Octave warns "matrix singular to machine precision".

### 6.4 Observables, measurement errors, shocks
- **varobs: 26**. 22 domestic: gdp_growth, c_growth, inve_growth, g_growth, infl_obs, r_obs, infl_n_obs,
  y_va_{n,m,z,h}_growth, ntwi_growth, z_x_growth, y_m_x_growth, hours_growth, infl_m_obs, pi_w_{n,m,z,h}_obs,
  infl_h_obs, inve_h_growth. 4 foreign: y_star_growth, infl_star_obs, r_star_obs, p_star_z_growth.
- **Measurement errors:** 30 explicit `varexo` (me*) with stderr 1, scaled in the equations by 30 calibrated
  parameters sd_me*. 8 scales are 0. Of the 22 non-zero, 20 sit on observed series and 2 (mex, mem) on unobserved ones.
  None is declared through Dynare's own measurement-error syntax.
- **Structural shocks: 28 declared.** eps_z_x is unused. eps_a has stderr 0, so aggregate technology is off. The five
  foreign-block standard deviations are fixed at stage-1 posterior means (6.3080, 0.9965, 0.0827, 0.0638, 0.1264).
  21 are estimated.

### 6.5 Estimation
- **estimated_params: 58**. 21 shock standard deviations, all **gamma** priors (not inverse gamma), written
  `stderr eps_x, init, gamma_pdf, mean, sd`. 37 parameters: 24 beta, 9 gamma, 4 normal. No p3/p4 shifts, no bounds,
  initial values given.
- `estimation(...)` in `estimate_model.mod`: `datafile='FourSectorData.xlsx', xls_sheet=DynareInput,
  xls_range=a1:ag101, prefilter=1, plot_priors=0, prior_trunc=0, mh_jscale=0.25, mode_compute=5, mode_check,
  mh_nblocks=2, mh_replic=50000`, plus a list of variables for posterior output.
- **Data:** `FourSectorData.xlsx`, sheet DynareInput. The first column holds dates (1992Q1 to 2016Q4) under an empty
  header; 32 series follow. The four wage series are `#N/A` until 1997Q3. **The stored estimation used 99 observations,
  1992Q2–2016Q4**: `options_.nobs = 99`, and all 26 smoothed observables equal the data minus their 99-quarter mean
  (non-missing values only) to within 1e-12. Within that sample the wage series have 22 missing quarters. The paper
  says the sample starts in 1992Q1. Why the first row was dropped is unknown; probably Dynare 4.4's spreadsheet reader
  (unverified).
- Software used: MATLAB 2015b, Dynare 4.4.0 (read-me).
- **Model size:** 143 endogenous (150 with auxiliaries), 58 exogenous (28 structural, 30 measurement errors), 189
  parameters, 26 observed series, 58 estimated parameters.

## 7. Reference results that can be matched exactly

**In the archive:**
- `Domestic Sector/multisector_mode.mat`: the posterior mode at full precision (`xparam1`, 58), the Hessian `hh`
  (58×58) and `parameter_names`.
- `Domestic Sector/multisector_results.mat` (Dynare 4.4 objects), including:
  - `M_.params` at the mode, with all steady-state values. These are the reference for the steady-state solve, e.g.
    lambda_ss 0.623380966852532, k_h_ss 17.480556399193645, omega_h 0.244675291371.
  - `oo_.MarginalDensity`: Laplace **-5139.2166564120**; modified harmonic mean **-5141.7721836111**.
  - `oo_.posterior.optimization.mode` and `.Variance` (equal to the mode file and inv(hh)).
  - `oo_.posterior.metropolis.mean` and `.Variance`, HPD bounds, medians, deciles, densities.
  - `oo_.dr` (ghx 150×80, ghu 150×58: the first-order solution, presumably at the mode, where `M_.params` sits).
  - `oo_.irfs` (40-quarter responses from the calibration-check `stoch_simul` calls), `oo_.SmoothedVariables` /
    `SmoothedShocks` / `UpdatedVariables` (99 periods).
  - Caution: the stored `M_.Sigma_e` has variance 1 for eps_r and eps_psi. The calibration-check file's `shocks` blocks
    overwrote the estimated values after estimation, so do not read the estimated shock sizes from there; use
    `oo_.posterior`.
  - `bayestopt_` (the priors as Dynare stored them), `estim_params_`, `options_`.
- **Derived from those** (script `stage4/scripts/mode_reference.py`): log posterior kernel at the mode **-5046.643162**;
  log prior density at the mode **-136.980736**; so log likelihood at the mode **-4909.662426**.
- **Checked by running Dynare 6.5** (Octave, scratch copy `stage4/dyn65_run/`: estimate_mod = 1, `nostrict`, the
  calibration-check include left out, data written to `reno_data.m`, `first_obs=2, nobs=99, prefilter=1,
  mode_compute=0, mode_file=multisector_mode, mh_replic=0`): Laplace **-5139.2166550595** (stored -5139.2166564120,
  difference 1.35e-6); log likelihood at the mode **-4909.6624245939** (derived -4909.662426); 4 seconds. The small
  difference is consistent with the steady-state solver's 1e-6 tolerance (not isolated).
- Mode against the paper (script `stage4/scripts/compare_mode_to_paper.py`): 57 of 58 posterior modes and all 58
  posterior means match Reno Tables 4 and 5 to two decimals. The exception: the eps_upsilon mode is 24.069 in the file
  and 24.01 in Table 5, probably a typo in the paper.
- `multisector/metropolis/metropolis.log`: 2 chains × 50,000 draws, acceptance about 0.30 per file, final log
  posterior per chain. The draw files (`multisector_mh*_blck*.mat`) and posterior IRF draws are in `reno.zip` and were
  not extracted. Stored `options_.mh_drop = 0.5`; the paper says it discards 20,000 of 100,000 draws.
- `Bayesian IRFs/bayes_irfs_four_sector.mat`: posterior IRF mean, median, deciles and HPD for 7 shocks.
  `bayes_irfs_three_sector.mat`: the same for the **intermediate three-sector model**, 3 shocks (cash rate, risk
  premium, commodity price). It is the only check available on the base-plus-revisions model, and an approximate one.
- `IRF_results.xlsx` (from IRF_construct.m), `HDecompNew.xlsx` (historical decomposition), `Scenarios.xlsx` (48 MB, not
  extracted), `rdp-2018-04-graph-data.xlsx` (figure data); foreign stage: `multisector_foreign_results.mat` and its
  MH log.

**Needs a Dynare run:** the log likelihood or posterior at points other than the mode; a fresh first-order solution at
other parameter values; the first-stage foreign estimation; posterior-mean responses beyond the stored ones. One-off
Octave runs at a given point take seconds. Full MH runs (2 × 50,000 draws) would take hours without the compiled
routines (not timed).

## 8. What mutatis's Dynare importer does not yet support (that this model uses)

1. **`@#include`.** `read_json` copies only the named `.mod` file into a temporary folder, so the six included files
   are not found. Verified: "Could not open variable_declarations.mod".
2. **Choosing a mode of the file.** The flags are `@#define`s inside the file; there is no way to pass them.
3. **Estimated shock standard deviations** (`stderr eps_x, ...` in `estimated_params`). Dynare's JSON keys them `var`,
   not `param`, so the importer crashes with `KeyError: 'param'` (verified). There is also no route for a prior on a
   shock's standard deviation other than a parameter named in the `shocks` block.
4. **Parameters computed by MATLAB code** (`ssvals(i)` from `Manualsteadystate`; `k_*_ss` set by native statements).
   Verified error: "the value of Al, ssvals(26), is not a number or an expression in other parameters' values". There
   are two ways in: take the numbers from the stored `M_.params`, or port `calcss_external.m` (38 equations with
   calibration targets, singular in the Al/xi_l scale) to mutatis's steady-state machinery.
5. **Model-local variables (`#`).** Ignored, so equations refer to undefined names. Verified at compile time:
   phi_pi_*, phi_pi_w_*, rstar_bar, phi_pi_star, phi_pi_m_star.
6. **`initval` that sets exogenous variables.** The importer puts them into the block's steady state, giving errors such
   as "'mey' is not an output of this block" (verified, 22 of them: initval lists 22 of the 30 error shocks).
7. **A declared but unused shock** (eps_z_x). The importer demands a standard deviation for it; Dynare 6.5 needs
   `nostrict`.
8. **The auxiliary-variable guard cannot fire.** `aux_vars` is empty at `json=parse`. This model needs a lag of 2 and
   six lagged shocks. mutatis's compiler has its own lag reducer (`mutatis/compile/lags.py` says it copies lagged
   shocks into the state); not tested on this model.
9. **`shocks` blocks inside MATLAB `if`s** (in the calibration-check file). The importer applies all of them in order,
   so eps_r and eps_psi would get stderr 1 after the real values.
10. **Measurement errors written as ordinary shocks times calibrated scales**, six of them MA(1). They import as
    structural shocks, not as data-link errors. That is fine for the likelihood, but they are not labelled as
    measurement errors.
11. **Data options are not read:** the `.xlsx` datafile with sheet and range; `prefilter=1` (demeaning by the sample
    mean of non-missing values; I found no demeaning option in mutatis); the sample (the stored run used 1992Q2–2016Q4).
    `Dataset.read_excel` needs a named date column, and this sheet's date column has an empty header (it would be found
    only as the text "None"). Missing values (NaN) are handled by mutatis's Kalman filter.
12. **`model(linear)` is not read.** Harmless here because every steady state is 0 (moderate confidence).
13. **The two-stage estimation and the steady state frozen at calibrated values** are file conventions, not syntax.
    An import that recomputes steady-state values at each draw will not reproduce Dynare's likelihood.

## 9. Other discrepancies found along the way

- **RSH Appendix A misprints:** A1 (h and b both used for habit), A7 (sign), A9 (E_t π_t for E_t π_t+1), A16 (r̂_z for
  k̂_z), probably A4 (missing −µ). Omissions: see Section 2.3.
- **Reno paper against code:**
  - policy rule (eq. 20 against R082);
  - Table 1 lists "ξ_wj = 0.25, elasticity of substitution between differentiated labour", but in the code xi_l_j are
    labour weights solved in the steady state (3.09, 9.21, 13.34, 9.18 in the stored run; only their products with Al
    are pinned down);
  - Table 1's A_H = 4.4177 is one arbitrary point on that ridge;
  - Table 5's eps_upsilon mode is 24.01 against 24.069;
  - the sample is 1992Q1 in the paper, 1992Q2 in the stored run;
  - MH burn-in.
- The read-me's flag name `define_irfs` against the code's `irfs`. The Forecasting copy of `multisector.mod` includes
  `dynamic_calibrations_postestimation_housing.mod` (when estimate_mod = 0), which is not in the archive.
- **mutatis, unrelated to the gate (found while probing):** in `mutatis/spec/checks.py`, `_block()`, the loop
  `for name, shock in block.shocks.items()` overwrites the function's `name` argument. Later messages from that
  function then label the block with the last shock's name (here "blocks.meinveh..." instead of
  "blocks.multisector..."). It affects labels only. Not changed.
