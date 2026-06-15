---
file_format: mystnb
kernelspec:
  name: python3
---

# Equations

This page collects the governing equations of the models implemented in
`pyqg-jax`. The equations describe what the code actually computes; the
notation matches the model classes in the {doc}`reference <reference>`.
For the original PyQG formulation see the
[PyQG equations page](https://pyqg.readthedocs.io/en/latest/equations.html).

## Common pseudospectral framework

Every model evolves a **potential vorticity** (PV) field $q$ that is
discretized in the horizontal on a doubly periodic grid and represented
by its two-dimensional Fourier transform $\hat q(k, l)$. Vertical
structure differs between models (layers, sheets, or a continuous
coordinate); writing the vertical index/argument as $i$, the prognostic
equation is the same flux-form advection in every case,

$$
\partial_t q_i + J(\psi_i, q_i) + U_i\,\partial_x q_i
  + Q_{y,i}\,\partial_x \psi_i = \mathcal{D}_i,
\qquad J(a,b) = a_x b_y - a_y b_x ,
$$

where $\psi$ is the streamfunction, $U_i$ is the background zonal flow,
$Q_{y,i}$ is the background PV gradient, and $\mathcal{D}_i$ collects
dissipation and forcing. The nonlinear advection is evaluated in
physical space and transformed back (a pseudospectral step); in spectral
form the tendency computed by the kernel is

$$
\partial_t \hat q_i = -\,\mathrm{i}k\,\widehat{(U_i + u_i)\,q_i}
  - \mathrm{i}l\,\widehat{v_i\,q_i}
  - \mathrm{i}k\,Q_{y,i}\,\hat\psi_i .
$$

The flow is non-divergent, so the velocities follow from the
streamfunction,

$$
u = -\partial_y \psi, \quad v = \partial_x \psi
\qquad\Longleftrightarrow\qquad
\hat u = -\mathrm{i}l\,\hat\psi, \quad \hat v = \mathrm{i}k\,\hat\psi .
$$

What distinguishes each model is the **inversion** relating $\hat\psi$ to
$\hat q$. In every model it is linear and diagonal in horizontal
wavenumber,

$$
\hat\psi_i(k,l) = \mathsf{a}_{ij}(k,l)\,\hat q_j(k,l) ,
$$

for a model-specific matrix/operator $\mathsf{a}$ (the inverse of the PV
operator). Writing $\kappa^2 = k^2 + l^2$ for the horizontal wavenumber
magnitude, the sections below give the PV operator, $\mathsf{a}$, $U_i$,
and $Q_{y,i}$ for each model.

## Two-layer QG model

{class}`~pyqg_jax.qg_model.QGModel` has two layers with potential
vorticity

$$
q_1 = \nabla^2\psi_1 + F_1(\psi_2 - \psi_1), \qquad
q_2 = \nabla^2\psi_2 + F_2(\psi_1 - \psi_2),
$$

with the stretching coefficients

$$
F_1 = \frac{f_0^2}{g'H_1}, \qquad
F_2 = \frac{f_0^2}{g'H_2} = \delta\,F_1, \qquad
\delta = \frac{H_1}{H_2},
$$

and reduced gravity $g' = g\,\Delta\rho/\rho_0$. In Fourier space the
inversion is the closed-form inverse of the $2\times 2$ system

$$
\begin{pmatrix}\hat\psi_1\\\hat\psi_2\end{pmatrix}
= -\frac{1}{\det}
\begin{pmatrix} \kappa^2 + F_2 & F_1 \\ F_2 & \kappa^2 + F_1 \end{pmatrix}
\begin{pmatrix}\hat q_1\\\hat q_2\end{pmatrix},
\qquad
\det = \kappa^2\,(\kappa^2 + F_1 + F_2),
$$

with the mean mode ($\kappa = 0$) set to zero. The background PV
gradients are

$$
Q_{y,1} = \beta + F_1(U_1 - U_2), \qquad
Q_{y,2} = \beta - F_2(U_1 - U_2).
$$

Bottom drag $r_{ek}\nabla^2\psi_2$ is applied to the lower layer.

## Layered QG model

{class}`~pyqg_jax.layered_model.LayeredModel` generalizes this to $n_z$
layers. Stacking the layers into a vector, the PV is

$$
\mathbf{q} = \nabla^2\boldsymbol{\psi} + \mathsf{S}\,\boldsymbol{\psi},
$$

where $\mathsf{S}$ is the tridiagonal **stretching matrix** built from
the layer thicknesses $H_i$ and reduced gravities $g'_i$,

$$
\mathsf{S}_{i,i-1} = \frac{f_0^2}{g'_{i-1} H_i}, \quad
\mathsf{S}_{i,i+1} = \frac{f_0^2}{g'_{i} H_i}, \quad
\mathsf{S}_{i,i} = -(\mathsf{S}_{i,i-1} + \mathsf{S}_{i,i+1}),
$$

(for $n_z = 2$ this reduces to the two-layer $F_1, F_2$ above). The
inversion solves, at each wavenumber, the linear system

$$
(\mathsf{S} - \kappa^2 \mathsf{I})\,\hat{\boldsymbol{\psi}} = \hat{\mathbf{q}},
$$

by a vectorized tridiagonal (Thomas) sweep for $n_z \ge 3$ and the
closed-form $2\times 2$ inverse for $n_z = 2$. The background PV gradient
is

$$
\mathbf{Q}_y = \beta - \mathsf{S}\,\mathbf{U}.
$$

## Equivalent barotropic model

{class}`~pyqg_jax.bt_model.BTModel` is a single layer with PV

$$
q = \nabla^2\psi - k_d^2\,\psi,
$$

where $k_d$ is the (optional) deformation wavenumber. The inversion is

$$
\hat\psi = -\frac{\hat q}{\kappa^2 + k_d^2},
$$

with $Q_y = \beta$ and $U$ the (optional) background flow. Setting
$k_d = 0$ gives the ordinary two-dimensional barotropic vorticity
equation.

## Surface QG model

{class}`~pyqg_jax.sqg_model.SQGModel` evolves surface buoyancy with zero
interior PV. The prognostic field is the surface buoyancy $b$, and the
inversion is the SQG Green's function for a semi-infinite lower domain,

$$
\hat\psi = \frac{f_0}{N}\,\frac{1}{\kappa}\,\hat q ,
$$

i.e. $\hat q = (N/f_0)\,\kappa\,\hat\psi$, the surface buoyancy
$b = f_0\,\partial_z\psi$ evaluated with the exponential decay
$\psi \propto e^{N\kappa z / f_0}$ into the interior.

## Two-Eady mixed-layer model

{class}`~pyqg_jax.callies_model.CalliesTwoEady` is the two-Eady
mixed-layer-instability model of Callies et al. (2016). It carries three
active buoyancy "sheets" — the surface, the mixed-layer base, and the
thermocline — each with its own stratification ($N_m$ in the mixed layer
of depth $H_m$, $N_t$ in the thermocline of depth $H_t$) and shear
($\Sigma_m, \Sigma_t$). The inversion is the closed-form inverse
(adjugate over determinant) of a symmetric tridiagonal $3\times 3$
operator $\mathsf{L}(\kappa)$ assembled from $\coth$/$\operatorname{csch}$
of $N\kappa H/f_0$ for each layer, evaluated with decaying exponentials
only to keep it well-conditioned. The background PV gradient is carried
entirely by the three sheets,

$$
\mathbf{Q}_y = \left(
\frac{f_0^2\,\Sigma_m}{N_m^2},\;
-f_0^2\!\left(\frac{\Sigma_m}{N_m^2} - \frac{\Sigma_t}{N_t^2}\right),\;
-\frac{f_0^2\,\Sigma_t}{N_t^2}
\right),
$$

with background flow $U = (u_1, u_1, u_2)$,
$u_1 = -\Sigma_m H_m$, $u_2 = u_1 - \Sigma_t H_t$. See the
{doc}`example <examples.callies>` for the reproduction of the paper's
linear and nonlinear results.

## Continuous-stratification model

{class}`~pyqg_jax.continuous_model.ContinuousQGModel` resolves a
continuous vertical coordinate $z \in [-H, 0]$ with a general
stratification $N^2(z)$, discretized by Chebyshev collocation. The
prognostic field holds the interior PV at the interior collocation points
and the surface/bottom buoyancy at the boundary points. The inversion
solves, at each horizontal wavenumber, the vertical boundary-value
problem

$$
\left[\partial_z\!\left(\frac{f_0^2}{N^2}\,\partial_z\right)
  - \kappa^2\right]\hat\psi = \hat q
\quad(\text{interior}), \qquad
f_0\,\partial_z\hat\psi = \hat b \quad(\text{at } z = 0, -H),
$$

with two Neumann (buoyancy) boundary conditions. The background PV
gradient is the continuous analogue of the layered expression,

$$
Q_y = \beta - \partial_z\!\left(\frac{f_0^2}{N^2}\,\partial_z\right) U
\quad(\text{interior}), \qquad
Q_y = -f_0\,\partial_z U \quad(\text{at the boundaries}),
$$

the boundary value being the thermal-wind surface buoyancy gradient. See
the {doc}`example <examples.continuous>`.

## Next-order (QG+1) model

{class}`~pyqg_jax.qgplus1_model.QGPlus1Model` carries the asymptotic
expansion one order further in the Rossby number $\varepsilon = U/(f_0 L)$
(Dù, Smith & Bühler 2024). The prognostic variables and the leading-order
inversion for $\Phi^0$ are exactly those of the continuous model above.
Three further Poisson problems, sharing the elliptic operator
$\mathcal{L} = N^2\nabla^2 + f_0^2\partial_{zz}$, give the first-order
corrections. For the Eady mean state ($U = \Lambda z$),

$$
\mathcal{L}(\Phi^1) = -f_0\!\left(
  -\frac{f_0^2}{N^2}(\Phi^0_{zz})^2 - |\nabla\Phi^0_z|^2
  + 2\Lambda\,\Phi^0_{yz}\right),
$$

$$
\mathcal{L}(F^1) = 2 f_0\!\left(J(\Phi^0_z, \Phi^0_x) + \Lambda\,\Phi^0_{xx}\right),
\qquad
\mathcal{L}(G^1) = 2 f_0\!\left(J(\Phi^0_z, \Phi^0_y) + \Lambda\,\Phi^0_{xy}\right),
$$

with homogeneous Neumann conditions for $\Phi^1$ and homogeneous
Dirichlet conditions for $F^1, G^1$. The velocities and buoyancy are then
reconstructed to first order,

$$
u = -\Phi^0_y - (\Phi^1_y + F^1_z), \qquad
v = \Phi^0_x + (\Phi^1_x - G^1_z),
$$

$$
w = F^1_x + G^1_y, \qquad
b = f_0\Phi^0_z + f_0\Phi^1_z + \frac{N^2}{f_0}\!\left(G^1_x - F^1_y\right).
$$

The vertical velocity $w$ is the self-consistent next-order field
(equivalent to the QG omega equation); the prognostic PV is advected by
the full three-dimensional next-order velocity, including the vertical
advection $w\,\partial_z q$. Because the source terms are *quadratic* in
$\Phi^0$, the model breaks the sign symmetry of QG and develops the
cyclone/anticyclone asymmetry of the submesoscale. See the
{doc}`example <examples.qgplus1>`.

## Time stepping

The state is advanced with a third-order Adams-Bashforth scheme
({class}`~pyqg_jax.steppers.AB3Stepper`), bootstrapped with a forward
Euler step followed by a second-order Adams-Bashforth step,

$$
\hat q^{n+1} = \hat q^{n} + \Delta t\left(
  a_1\,\partial_t\hat q^{\,n} + a_2\,\partial_t\hat q^{\,n-1}
  + a_3\,\partial_t\hat q^{\,n-2}\right),
$$

with $(a_1, a_2, a_3) = (23/12, -16/12, 5/12)$. A fourth-order
Runge-Kutta scheme ({class}`~pyqg_jax.steppers.RK4Stepper`) with a larger
stability region is also available, as is a forward Euler stepper. The
steppers are generic over the model state and compose with
{func}`jax.jit`, {func}`jax.vmap`, and {func}`jax.grad`.

## Dissipation

Small-scale enstrophy is removed by a spectral exponential filter applied
once per step (in `postprocess_state`),

$$
\hat q \mapsto \Theta(k, l)\,\hat q, \qquad
\Theta =
\begin{cases}
1 & |\mathbf{k}_\Delta| \le k_c, \\[4pt]
\exp\!\big(-c\,(|\mathbf{k}_\Delta| - k_c)^4\big) & |\mathbf{k}_\Delta| > k_c,
\end{cases}
$$

where $|\mathbf{k}_\Delta| = \sqrt{(k\,\Delta x)^2 + (l\,\Delta y)^2}$,
$k_c = 0.65\pi$, and $c$ is the filter amplitude (``filterfac``).
Optional linear bottom drag $r_{ek}$ acts on the lowest layer.

## Linear stability analysis

The base models provide a
{meth}`~pyqg_jax.qg_model.QGModel.stability_analysis` method that
linearizes the dynamics about the background state. At each wavenumber it
solves the eigenvalue problem for normal modes
$\hat\theta \propto e^{-\mathrm{i}\omega t}$,

$$
k\,\big(\mathsf{U} + \mathsf{Q}_y\,\mathsf{a}\big)\,\tilde\theta
  = \omega\,\tilde\theta,
$$

where $\mathsf{U} = \operatorname{diag}(U_i)$,
$\mathsf{Q}_y = \operatorname{diag}(Q_{y,i})$, and $\mathsf{a}$ is the
model's inversion matrix. The growth rate is $\operatorname{Im}(\omega)$
and the phase speed is $\operatorname{Re}(\omega)/k$. This recovers, for
example, the analytic Eady and Charney dispersion relations for the
continuous model (see its {doc}`example <examples.continuous>`).
