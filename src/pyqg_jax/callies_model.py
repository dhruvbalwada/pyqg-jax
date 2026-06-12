# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


"""An implementation of :class:`pyqg.CalliesTwoEady`."""

__all__ = ["CalliesTwoEady"]


import jax
import jax.numpy as jnp
from . import _model, _utils, state as _state


@_utils.register_pytree_class_attrs(
    children=["Nm", "Nt", "Hm", "Ht", "Sm", "St", "nu", "hypodiff", "dt"],
    static_attrs=["nun", "use_dealias_filter"],
)
class CalliesTwoEady(_model.Model):
    r"""Two-Eady mixed-layer-instability model of Callies, Flierl,
    Ferrari & Fox-Kemper (2016, JFM, doi:10.1017/jfm.2015.700).

    A weakly stratified mixed layer (buoyancy frequency :math:`N_m`,
    depth :math:`H_m`, zonal shear :math:`\Lambda_m`) sits on top of a
    strongly stratified thermocline (:math:`N_t`, thickness
    :math:`H_t`, shear :math:`\Lambda_t`). Both layers have zero
    interior PV, so the dynamics is fully described by three PV sheets
    at the surface, the layer interface, and the bottom.

    See also :class:`pyqg.CalliesTwoEady`.

    .. versionadded:: 0.9.0

    Parameters
    ----------
    nx : int, optional
        Number of grid points in the `x` direction.

    ny : int, optional
        Number of grid points in the `y` direction. Defaults to `nx`.

    L : float, optional
        Domain length in the `x` direction. Units: :math:`\mathrm{m}`.

    W : float, optional
        Domain length in the `y` direction. Defaults to `L`.
        Units: :math:`\mathrm{m}`.

    rek : float, optional
        Linear drag in lower layer. Units: :math:`\mathrm{sec}^{-1}`.
        This model has no bottom Ekman friction by default.

    filterfac : float, optional
        Amplitude of the spectral spherical cutoff filter. Only used
        when `use_dealias_filter` is :pycode:`True`, in which case the
        standard PyQG cutoff is composed with the hyper/hypoviscous
        dissipation filter.

    f : float, optional
        Coriolis parameter. Units: :math:`\mathrm{sec}^{-1}`.

    g : float, optional

    Nm : float, optional
        Mixed layer buoyancy frequency. Units: :math:`\mathrm{sec}^{-1}`.

    Nt : float, optional
        Thermocline buoyancy frequency. Units: :math:`\mathrm{sec}^{-1}`.

    Hm : float, optional
        Mixed layer depth. Units: :math:`\mathrm{m}`.

    Ht : float, optional
        Thermocline thickness. Units: :math:`\mathrm{m}`.

    Sm : float, optional
        Mixed layer zonal shear :math:`\Lambda_m`.
        Units: :math:`\mathrm{sec}^{-1}`.

    St : float, optional
        Thermocline zonal shear :math:`\Lambda_t`.
        Units: :math:`\mathrm{sec}^{-1}`.

    nu : float, optional
        Hyperviscosity coefficient :math:`\nu` of the operator
        :math:`\nu(-\nabla^2)^n`. Units: :math:`\mathrm{m}^{2n}\
        \mathrm{sec}^{-1}`. If :pycode:`None` (default, requires
        :pycode:`nun == 10`), the value :math:`2.5 \times 10^{46}`
        used by Callies et al. (2016) at :math:`512^2` on a 500 km
        domain is rescaled to the model grid spacing so that damping
        at the grid scale is resolution independent.

    nun : int, optional
        Hyperviscosity order :math:`n` (default 10, i.e.
        :math:`\nabla^{20}`). This is a *static* attribute (not a
        traced pytree child).

    hypodiff : float, optional
        Hypoviscosity coefficient :math:`r` of the operator
        :math:`r\nabla^{-2}`, arresting the inverse cascade.
        Units: :math:`\mathrm{m}^{-2}\ \mathrm{sec}^{-1}`.
        Set to 0 to disable.

    use_dealias_filter : bool, optional
        Whether to compose PyQG's standard exponential spectral cutoff
        filter (controlled by `filterfac`) with the dissipation
        filter. Defaults to :pycode:`True`, standing in for the
        3/2-rule dealiasing of the original authors' code by absorbing
        aliased enstrophy near the grid scale. This is a *static*
        attribute (not a traced pytree child).

    dt : float
        Numerical time step used to build the dissipation filter.
        Units: :math:`\mathrm{sec}`.

        .. warning:: This **must** match the `dt` of the time stepper
           used with this model (the same coupling exists in PyQG,
           where the model owns `dt`). The dissipation is applied
           implicitly as a multiplicative spectral factor
           :math:`e^{-\nu \kappa^{2n} \Delta t} e^{-r \kappa^{-2}
           \Delta t}` once per time step, exactly as in the original
           authors' code.

    precision : Precision, optional
        Precision of model computation. Selects dtype of state values.

    Attributes
    ----------
    Ubg : jax.Array
        The background velocity for this model.

    Note
    ----
    This model internally uses 64-bit floating point values for part
    of its computation *regardless* of the chosen :class:`precision
    <pyqg_jax.state.Precision>`. (In particular the hyperviscous
    filter underflows in 32-bit arithmetic.)

    Make sure that JAX has `64-bit precision enabled
    <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`__.
    """

    def __init__(
        self,
        *,
        # grid size parameters
        nx=64,
        ny=None,
        L=5e5,
        W=None,
        # friction parameters
        rek=0.0,
        filterfac=23.6,
        # constants
        f=1e-4,
        g=9.81,
        # Additional model parameters
        Nm=2e-3,
        Nt=8e-3,
        Hm=100.0,
        Ht=400.0,
        Sm=1e-4,
        St=1e-4,
        nu=None,
        nun=10,
        hypodiff=1e-16,
        dt,
        use_dealias_filter=True,
        # Precision choice
        precision=_state.Precision.SINGLE,
    ):
        super().__init__(
            nz=3,
            nx=nx,
            ny=ny,
            L=L,
            W=W,
            rek=rek,
            filterfac=filterfac,
            f=f,
            g=g,
            precision=precision,
        )
        self.Nm = Nm
        self.Nt = Nt
        self.Hm = Hm
        self.Ht = Ht
        self.Sm = Sm
        self.St = St
        self.nun = nun
        if nu is None:
            if nun != 10:
                raise ValueError("provide nu explicitly when nun != 10")
            # paper value, rescaled so damping at the grid scale is the
            # same as at 512^2 on a 500 km domain
            nu = 2.5e46 * ((512.0 / 5e5) / (self.nx / self.L)) ** (2 * nun)
        self.nu = nu
        self.hypodiff = hypodiff
        self.dt = dt
        self.use_dealias_filter = use_dealias_filter

    def create_initial_state(self, key):
        """Create a new initial state with random initialization.

        The initial condition is small white noise in the PV sheets,
        with zero mean, following Callies et al. (2016).

        Parameters
        ----------
        key : jax.random.key
            The PRNG state used as the random key for initialization.

        Returns
        -------
        PseudoSpectralState
            The new state with random initialization.
        """
        q = 1e-4 * jax.random.uniform(
            key,
            shape=(self.nz, self.ny, self.nx),
            dtype=self.precision.dtype_real,
        )
        q = q - q.mean(axis=(-2, -1), keepdims=True)
        return super().create_initial_state().update(q=q)

    @property
    def Hi(self):
        # trapezoidal weights attributing half of each layer to the
        # bounding PV sheets; used only by depth-weighted diagnostics
        return jnp.array(
            [self.Hm / 2, (self.Hm + self.Ht) / 2, self.Ht / 2],
            dtype=self.precision.dtype_real,
        )

    @property
    def H(self):
        return self.get_grid().H

    @property
    def Ubg(self):
        # mean zonal flow at surface, interface, bottom (thermal wind,
        # referenced to zero at the surface)
        u1 = -self.Sm * self.Hm
        u2 = u1 - self.St * self.Ht
        return jnp.array([jnp.zeros_like(u1), u1, u2], dtype=self.precision.dtype_real)

    @property
    def Qy(self):
        # meridional gradients of the conserved quantities
        # (cf. Callies et al. 2016, eq. 2.12)
        f2 = self.f**2
        return jnp.array(
            [
                f2 * self.Sm / self.Nm**2,
                -f2 * (self.Sm / self.Nm**2 - self.St / self.Nt**2),
                -f2 * self.St / self.Nt**2,
            ],
            dtype=self.precision.dtype_real,
        )

    @property
    def ikQy(self):
        return jnp.expand_dims(self.Qy, (-1, -2)) * 1j * self.k

    @property
    def ilQx(self):
        return 0.0

    @property
    def filtr(self):
        # Implicit hyper- and hypoviscosity, mirroring the dissipation
        # step of the authors' code: after each timestep the PV sheets
        # are multiplied by exp(-nu kappa^{2n} dt) * exp(-r kappa^{-2} dt).
        # Computed in float64: kappa^{2n} underflows in float32.
        wv2 = self.wv2.astype(jnp.float64)
        f64_dt = jnp.float64(self.dt)
        filtr = jnp.exp(-jnp.float64(self.nu) * wv2**self.nun * f64_dt)
        # wv2i is zero at kappa = 0, so the mean is untouched (and the
        # factor is exactly 1 when hypodiff is 0)
        wv2i = self.wv2i.astype(jnp.float64)
        filtr = filtr * jnp.exp(-jnp.float64(self.hypodiff) * wv2i * f64_dt)
        if self.use_dealias_filter:
            # compose PyQG's standard exponential cutoff (cf.
            # _model.Model.filtr), standing in for the 3/2-rule
            # dealiasing of Callies et al.'s code
            filtr = filtr * _model.Model.filtr.fget(self).astype(jnp.float64)
        return filtr.astype(self.precision.dtype_real)

    def _apply_a_ph(self, state):
        # Invert qh for ph using the closed-form (adjugate / det)
        # inverse of the symmetric tridiagonal inversion matrix
        #     L = [[la, lb, 0], [lb, lc, ld], [0, ld, le]]
        # (cf. Callies et al. 2016, eq. 2.11), written out by hand so
        # that it can be traced under jit with traced parameters.
        f64_f = jnp.float64(self.f)
        f64_nm = jnp.float64(self.Nm)
        f64_nt = jnp.float64(self.Nt)
        f64_hm = jnp.float64(self.Hm)
        f64_ht = jnp.float64(self.Ht)
        wv = self.wv.astype(jnp.float64)
        # avoid division by zero; (0, 0) is zeroed below
        kh = jnp.where(wv == 0, 1.0, wv)
        mum = f64_nm * kh * f64_hm / f64_f
        mut = f64_nt * kh * f64_ht / f64_f
        # coth/csch from decaying exponentials only: no overflow at
        # large kappa H N / f (where csch -> 0 and coth -> 1), so
        # gradients stay finite
        expm = jnp.exp(-2 * mum)
        expt = jnp.exp(-2 * mut)
        cothm = (1 + expm) / (1 - expm)
        cschm = 2 * jnp.exp(-mum) / (1 - expm)
        cotht = (1 + expt) / (1 - expt)
        cscht = 2 * jnp.exp(-mut) / (1 - expt)
        la = -f64_f * kh * cothm / f64_nm
        lb = f64_f * kh * cschm / f64_nm
        lc = -f64_f * kh * cothm / f64_nm - f64_f * kh * cotht / f64_nt
        ld = f64_f * kh * cscht / f64_nt
        le = -f64_f * kh * cotht / f64_nt
        det = la * (lc * le - ld**2) - lb**2 * le
        qh = state.qh.astype(jnp.complex128)
        ph = (
            jnp.stack(
                [
                    (lc * le - ld**2) * qh[0] - (lb * le) * qh[1] + (lb * ld) * qh[2],
                    -(lb * le) * qh[0] + (la * le) * qh[1] - (la * ld) * qh[2],
                    (lb * ld) * qh[0] - (la * ld) * qh[1] + (la * lc - lb**2) * qh[2],
                ]
            )
            / det
        )
        # the inversion is not defined at kappa = 0 (zero-mean fields)
        ph = jnp.where(wv == 0, 0.0, ph)
        return ph.astype(self.precision.dtype_complex)
