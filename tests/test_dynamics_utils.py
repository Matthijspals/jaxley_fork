# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, value_and_grad

import jaxley as jx
import jaxley.optimize.transforms as jt
from jaxley.channels import Leak
from jaxley.channels.hh import HH
from jaxley.connect import fully_connect
from jaxley.integrate import add_stimuli, build_init_and_step_fn
from jaxley.pumps import CaNernstReversal, CaPump
from jaxley.synapses.ionotropic import IonotropicSynapse
from jaxley.synapses.test import TestSynapse
from jaxley.utils.dynamics import build_dynamic_state_utils


def test_cycle_consistency():
    """Ensure that ravel/unravel of state vectors is consistent"""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 3)
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.insert(HH())
    cell.insert(Leak())
    cell.insert(CaNernstReversal())
    cell.insert(CaPump())
    cell.to_jax()

    init_fn, step_fn = build_init_and_step_fn(cell)
    (
        remove_observables,
        add_observables,
        flatten,
        unflatten,
        restore_structure,
    ) = build_dynamic_state_utils(cell)
    all_states, all_params = init_fn([])
    dynamic_states = flatten(remove_observables(all_states))
    restored = add_observables(unflatten(dynamic_states), all_params, delta_t=0.025)
    reraveled = flatten(remove_observables(restored))
    assert np.allclose(reraveled, dynamic_states)

    # restore_structure restores padding/branchpoints but does not compute currents.
    structured = restore_structure(unflatten(dynamic_states))
    assert "i_HH" not in structured
    assert "i_Leak" not in structured
    assert "i_HH" in restored
    assert np.allclose(structured["v"], restored["v"])
    assert np.allclose(structured["HH_m"], restored["HH_m"])


def test_restore_structure_step_matches_add_observables():
    """For HH, step after restore_structure matches step after add_observables."""
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=4), parents=[-1])
    cell.insert(HH())
    delta_t = 0.025
    t_max = 1.0
    n_steps = int(t_max / delta_t)
    cell.branch(0).comp(0).stimulate(jx.step_current(0.1, 0.5, 0.05, delta_t, t_max))
    cell.to_jax()

    init_fn, step_fn = build_init_and_step_fn(cell)
    (
        remove_observables,
        add_observables,
        flatten,
        unflatten,
        restore_structure,
    ) = build_dynamic_state_utils(cell)

    all_states, all_params = init_fn([])
    dynamic_states = flatten(remove_observables(all_states))
    externals = cell.externals.copy()
    external_inds = cell.external_inds.copy()

    dyn_add = dynamic_states
    dyn_restore = dynamic_states
    for step in range(n_steps):
        externals_now = {k: externals[k][:, step] for k in externals}

        all_states_add = add_observables(unflatten(dyn_add), all_params, delta_t)
        all_states_add = step_fn(
            all_states_add, all_params, externals_now, external_inds, delta_t=delta_t
        )
        dyn_add = flatten(remove_observables(all_states_add))

        all_states_restore = restore_structure(unflatten(dyn_restore))
        assert "i_HH" not in all_states_restore
        all_states_restore = step_fn(
            all_states_restore, all_params, externals_now, external_inds, delta_t=delta_t
        )
        dyn_restore = flatten(remove_observables(all_states_restore))

        assert np.allclose(dyn_add, dyn_restore)


@pytest.mark.parametrize(
    "branchpoint", [True, False], ids=["branchpoint", "no_branchpoint"]
)
def test_build_step_dynamics_fn_branchpoints(branchpoint):
    """Check state vector length changes with/without branchpoints"""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 8)
    parents = [-1, 0, 0] if branchpoint else [-1]
    cell = jx.Cell(branch, parents=parents)
    cell.branch(0).insert(HH())
    cell.insert(Leak())
    cell.to_jax()

    # get states and unflatten functions
    init_fn, step_fn = build_init_and_step_fn(cell)
    (
        remove_observables,
        add_observables,
        flatten,
        unflatten,
        restore_structure,
    ) = build_dynamic_state_utils(cell)
    all_states, all_params = init_fn([])
    dynamic_states = flatten(remove_observables(all_states))

    # check lengths
    tree = unflatten(dynamic_states)
    structured = restore_structure(tree)
    full_tree = add_observables(tree, all_params, delta_t=0.025)
    v_len_full = len(full_tree["v"])
    v_len = len(tree["v"])
    v_len_structured = len(structured["v"])
    i_hh_len = len(full_tree["i_HH"])
    i_hh_m_nonzero = np.count_nonzero(~np.isnan(full_tree["HH_m"]))

    assert "i_HH" not in structured
    assert v_len_structured == v_len_full

    if branchpoint:
        assert v_len == 24
        assert v_len_full == 25  # should be n_branches * ncomp_per_branch + 1
        assert i_hh_len == 25
        assert i_hh_m_nonzero == 9
    else:
        assert v_len == 8
        assert v_len_full == 8
        assert i_hh_len == 8
        assert i_hh_m_nonzero == 8


def test_jit_and_grad_network():
    """A full gradient step on a network.

    This implicitly tests also:
    1) Whether the `step_dynamics` function can be jitted.
    2) Whether .make_trainable() works.
    3) Whether .data_set and `param_state` work.
    4) Whether `restore_structure` and `add_observables` give matching grads for Leak.
    """
    branch = jx.Branch()
    cell = jx.Cell(branch, [-1, 0])
    cell = jx.Cell()
    net = jx.Network([cell for _ in range(5)], vectorize_cells=True)
    net.insert(Leak())

    fully_connect(net.cell(range(2)), net.cell(range(3)), TestSynapse())
    fully_connect(net.cell(range(3, 5)), net.cell(range(2, 5)), IonotropicSynapse())
    # make some parameters trainable
    net.make_trainable("Leak_gLeak")
    net.make_trainable("v")
    params = net.get_parameters()

    # Define parameter transform and apply it to the parameters.
    transform = jx.ParamTransform(
        [
            {"Leak_gLeak": jt.SigmoidTransform(0.00001, 0.0002)},
            {"v": jt.SigmoidTransform(-100, -30)},
        ]
    )
    opt_params = transform.inverse(params)
    params = transform.forward(opt_params)
    net.to_jax()

    # add some inputs
    externals = net.externals.copy()
    external_inds = net.external_inds.copy()
    current = jx.step_current(
        i_delay=1.0, i_dur=2.0, i_amp=0.08, delta_t=0.025, t_max=0.075
    )
    data_stimuli = None
    data_stimuli = net.branch(0).comp(0).data_stimulate(current, data_stimuli)
    externals, external_inds = add_stimuli(externals, external_inds, data_stimuli)

    def get_externals_now(externals, step):
        externals_now = {}
        for key in externals.keys():
            externals_now[key] = externals[key][:, step]
        return externals_now

    target_voltage = -50.0
    state_idx = 0

    init_fn, step_fn = build_init_and_step_fn(net)
    (
        remove_observables,
        add_observables,
        flatten,
        unflatten,
        restore_structure,
    ) = build_dynamic_state_utils(net)

    def init_dynamics(params, param_state):
        all_states, all_params = init_fn(params, None, param_state)
        dynamic_states = flatten(remove_observables(all_states))
        return dynamic_states, all_params

    def make_step_dynamics(rebuild):
        @jit
        def step_dynamics(
            dynamic_states, all_params, externals, external_inds, delta_t=0.025
        ):
            all_states = rebuild(unflatten(dynamic_states), all_params, delta_t)
            all_states = step_fn(
                all_states, all_params, externals, external_inds, delta_t=delta_t
            )
            dynamic_states = flatten(remove_observables(all_states))
            return dynamic_states

        return step_dynamics

    def make_loss(step_dynamics):
        def loss(opt_params, pstate_params):
            pstate = net.data_set("radius", pstate_params, None)

            params = transform.forward(opt_params)
            dynamic_states, all_params = init_dynamics(params, pstate)
            dynamic_states_list = [dynamic_states]

            # Simulate the model
            for step in range(3):
                # Get inputs at this time step
                externals_now = get_externals_now(externals, step)
                # Step the ODE
                dynamic_states = step_dynamics(
                    dynamic_states,
                    all_params,
                    externals_now,
                    external_inds,
                    delta_t=0.025,
                )
                # Store the state
                dynamic_states_list.append(dynamic_states)
            # Compute the loss at the last time step
            loss = jnp.mean((dynamic_states_list[-1][state_idx] - target_voltage) ** 2)
            return loss

        return loss

    def rebuild_add(tree, all_params, delta_t):
        return add_observables(tree, all_params, delta_t=delta_t)

    def rebuild_restore(tree, all_params, delta_t):
        return restore_structure(tree)

    results = {}
    for name, rebuild in [("add", rebuild_add), ("restore", rebuild_restore)]:
        step_dynamics = make_step_dynamics(rebuild)
        loss = make_loss(step_dynamics)
        value, gradient = value_and_grad(loss, argnums=(0, 1))(
            opt_params, jnp.asarray(0.1)
        )
        opt_params_grads = gradient[0]
        pstate_params_grads = gradient[1]
        assert np.all(abs(opt_params_grads[0]["Leak_gLeak"]) > 0)
        assert np.all(abs(opt_params_grads[1]["v"]) > 0)
        assert np.all(abs(pstate_params_grads) > 0)
        results[name] = (value, opt_params_grads, pstate_params_grads)

    # Leak does not read currents, so both rebuild paths must match.
    assert np.allclose(results["add"][0], results["restore"][0])
    assert np.allclose(
        results["add"][1][0]["Leak_gLeak"], results["restore"][1][0]["Leak_gLeak"]
    )
    assert np.allclose(results["add"][1][1]["v"], results["restore"][1][1]["v"])
    assert np.allclose(results["add"][2], results["restore"][2])
