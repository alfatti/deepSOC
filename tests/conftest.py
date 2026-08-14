"""Session-scoped solves shared across the milestone-2 gate tests."""
import pytest

import mfoptmm as mf


@pytest.fixture(scope="session")
def body():
    """Body variant (c_rho = 1) on BBG's replication grid — the paper's own
    problem, with the full quote schedule stored for the policy gates."""
    spec = mf.bbg_example(variant="body", profile="replication")
    return spec, mf.hjb_solve(spec, store_quotes=True)


@pytest.fixture(scope="session")
def a1():
    """A.1 variant (c_rho = 0.75) — the study object."""
    spec = mf.bbg_example(variant="a1", profile="replication")
    return spec, mf.hjb_solve(spec, store_quotes=False)


@pytest.fixture(scope="session")
def collapsed():
    """Remark-6 configuration: a^P = a^Q (kappaQ := kappaP, thetaQ := thetaP),
    body penalty — the exact nu-collapse and bid/ask symmetry test bed."""
    base = mf.bbg_example(variant="body", profile="replication")
    spec = base._replace(
        market=base.market._replace(kappaQ=base.market.kappaP,
                                    thetaQ=base.market.thetaP))
    return spec, mf.hjb_solve(spec, store_quotes=True)


@pytest.fixture(scope="session")
def fine(body):
    """Body variant on a spatially refined Vpi grid (n_V = 161): the upper
    edge of the E5 bracket and the matched-resolution myopic reference."""
    from mfoptmm.config import VpiGrid
    spec = body[0]._replace(vpi_grid=VpiGrid(n=161))
    return spec, mf.hjb_solve(spec, store_quotes=True)
