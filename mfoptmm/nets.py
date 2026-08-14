"""Per-block control networks for the BCR trainer.

One hidden layer, width r, tanh, 1/r output scaling (mean-field
parameterisation: each neuron is a particle theta_i = (a_i, b_i, c_i) in
R^{2+1+41}). Outputs are *offsets from the reference control* in natural
units, so a zero-output net is exactly the reference policy:

    delta^{i,j}(x) = delta_myopic^i + u^{i,j}(x) / b^i     (quote offsets in
                                                            spread units 1/b)
    eta(x)         = (|rho| xi / 2) Vbar * u_eta(x)         (so the exact hedge
                                                            eta* = -(rho xi/2) Vpi
                                                            = +0.05 Vpi here
                                                            is u_eta = +Vpi/Vbar)

Inputs are normalised: x = [ (nu - nu0) / nu_scale, Vpi / Vbar ].

`c` is initialised at 0.01-scale noise (near-zero: the policy starts at the
reference, and gradients reach (a, b) as soon as c moves; MFLD noise breaks
the exact-zero saddle regardless).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ProblemSpec
from .model import myopic_quotes

NU_SCALE = 0.009  # half-width of the nu grid


class BlockNet(NamedTuple):
    a: jnp.ndarray   # (r, 2)
    b: jnp.ndarray   # (r,)
    c: jnp.ndarray   # (r, 41): 40 quote channels (N, 2) + 1 eta channel


def init_net(key, r: int) -> BlockNet:
    ka, kb, kc = jax.random.split(key, 3)
    return BlockNet(a=jax.random.normal(ka, (r, 2)) / jnp.sqrt(2.0),
                    b=0.5 * jax.random.normal(kb, (r,)),
                    c=0.01 * jax.random.normal(kc, (r, 41)))


def init_stack(key, K: int, r: int) -> BlockNet:
    """K independent block nets stacked on a leading axis: fields (K, r, ...)."""
    nets = [init_net(jax.random.fold_in(key, k), r) for k in range(K)]
    return BlockNet(*[jnp.stack([getattr(n, f) for n in nets])
                      for f in BlockNet._fields])


def raw_output(net: BlockNet, x):
    """x (B, 2) -> (B, 41): (1/r) sum_i c_i tanh(a_i . x + b_i)."""
    h = jnp.tanh(x @ net.a.T + net.b[None, :])          # (B, r)
    return h @ net.c / net.c.shape[0]                    # (B, 41)


class ControlMaps(NamedTuple):
    """Static pieces turning raw outputs into controls."""
    delta_ref: jnp.ndarray   # (N,) myopic quotes
    inv_b: jnp.ndarray       # (N,) spread unit 1/b^i
    eta_scale: float
    nu0: float
    Vbar: float


def control_maps(spec: ProblemSpec) -> ControlMaps:
    m, r = spec.market, spec.risk
    return ControlMaps(delta_ref=myopic_quotes(spec.book),
                       inv_b=1.0 / spec.book.b,
                       eta_scale=abs(m.rho) * m.xi / 2.0 * r.Vbar,
                       nu0=m.nu0, Vbar=r.Vbar)


def controls(net: BlockNet, cm: ControlMaps, nu, Vpi):
    """Batched controls at states (nu, Vpi) (B,): (delta (B, N, 2), eta (B,))."""
    x = jnp.stack([(nu - cm.nu0) / NU_SCALE, Vpi / cm.Vbar], axis=-1)
    out = raw_output(net, x)                              # (B, 41)
    u = out[:, :40].reshape(-1, cm.delta_ref.size, 2)
    delta = cm.delta_ref[None, :, None] + u * cm.inv_b[None, :, None]
    eta = cm.eta_scale * out[:, 40]
    return delta, eta


def materialise_schedule(stack: BlockNet, cm: ControlMaps, spec: ProblemSpec):
    """Evaluate the block nets on the PDE grid and repeat per substep:
    returns (quotes (M, n_nu, n_V, N, 2), eta (M, n_nu, n_V)) for the exact
    arbiter (policy_value)."""
    from .hjb import make_ops
    ops = make_ops(spec)
    nn, nv = ops.nu.size, ops.V.size
    nu_g = jnp.repeat(ops.nu, nv)
    V_g = jnp.tile(ops.V, nn)

    def per_block(k):
        net = jax.tree_util.tree_map(lambda f: f[k], stack)
        d, e = controls(net, cm, nu_g, V_g)
        return (d.reshape(nn, nv, cm.delta_ref.size, 2), e.reshape(nn, nv))

    ds, es = jax.lax.map(per_block, jnp.arange(stack.a.shape[0]))
    s = spec.time_grid.substeps_per_block
    return jnp.repeat(ds, s, axis=0), jnp.repeat(es, s, axis=0)
