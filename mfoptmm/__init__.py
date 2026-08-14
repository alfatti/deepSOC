"""mfoptmm: mean-field learned quoting vs exact HJB on the BBG A.1 options
market-making problem.

Float64 is enabled at import time: the exact solver and the identity gates in
this package are meaningless in float32. Import mfoptmm before creating any
JAX arrays elsewhere in a process.
"""
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .model import (  # noqa: E402
    Market,
    Book,
    Risk,
    a_P,
    a_Q,
    carry,
    penalty_reduced,
    penalty_hedge,
    eta_star,
    intensity,
    intensity_inv,
    wright_omega,
    H,
    H_p,
    H_pp,
    delta_star,
    myopic_quotes,
    post_trade_ok,
)
from .instruments import (  # noqa: E402
    PAPER_TABLE,
    paper_book,
    pricer_book,
    heston_call,
    heston_price_and_vega,
    bs_call,
    implied_vol,
    verify_paper_table,
)
from .config import (  # noqa: E402
    NuGrid,
    VpiGrid,
    TimeGrid,
    ProblemSpec,
    stability_rates,
    derive_time_grid,
    bbg_example,
)
from .hjb import (  # noqa: E402
    HJBSolution,
    hjb_solve,
    make_ops,
    at_state,
    interp_axis,
)
from .simulate import (  # noqa: E402
    SimResult,
    simulate_schedule,
    refine_schedule,
    refined_spec,
    draw_vpi0,
)
from .nets import (  # noqa: E402
    BlockNet,
    ControlMaps,
    control_maps,
    controls,
    init_stack,
    materialise_schedule,
)
from .train import (  # noqa: E402
    TrainConfig,
    TrainedPolicy,
    train,
    train_block,
    rollout_value,
    rollin_states,
    full_objective,
    objective_scale,
)
from .policy_eval import (  # noqa: E402
    PolicyValue,
    policy_value,
    repeat_block_start,
    constant_policy,
    myopic_policy,
)

__all__ = [
    "Market", "Book", "Risk",
    "a_P", "a_Q", "carry", "penalty_reduced", "penalty_hedge", "eta_star",
    "intensity", "intensity_inv", "wright_omega",
    "H", "H_p", "H_pp", "delta_star", "myopic_quotes", "post_trade_ok",
    "PAPER_TABLE", "paper_book", "pricer_book", "heston_call",
    "heston_price_and_vega", "bs_call", "implied_vol", "verify_paper_table",
    "NuGrid", "VpiGrid", "TimeGrid", "ProblemSpec",
    "stability_rates", "derive_time_grid", "bbg_example",
    "HJBSolution", "hjb_solve", "make_ops", "at_state", "interp_axis",
    "PolicyValue", "policy_value", "repeat_block_start", "constant_policy",
    "myopic_policy",
    "SimResult", "simulate_schedule", "refine_schedule", "refined_spec",
    "draw_vpi0",
    "BlockNet", "ControlMaps", "control_maps", "controls", "init_stack",
    "materialise_schedule",
    "TrainConfig", "TrainedPolicy", "train", "train_block", "rollout_value",
    "rollin_states", "full_objective", "objective_scale",
]
