# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy

import jax
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.modules.base import View
from jaxley.synapses import TestSynapse
from jaxley.utils.cell_utils import loc_of_index, local_index_of_loc
from jaxley.utils.misc_utils import cumsum_leading_zero
from jaxley.utils.solver_utils import JaxleySolveIndexer


def test_getitem():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(3)], parents=jnp.asarray([-1, 0, 0]))
    net = jx.Network([cell for _ in range(3)])

    # test API equivalence
    assert all(net.cell(0).branch(0).show() == net[0, 0].show())
    assert all(net.cell([0, 1]).branch(0).show() == net[[0, 1], 0].show())
    assert all(net.cell(0).branch([0, 1]).show() == net[0, [0, 1]].show())
    assert all(net.cell([0, 1]).branch([0, 1]).show() == net[[0, 1], [0, 1]].show())

    assert all(net.cell(0).branch(0).comp(0).show() == net[0, 0, 0].show())
    assert all(cell.branch(0).comp(0).show() == cell[0, 0].show())
    assert all(cell.branch(0).show() == cell[0].show())

    # test indexing of comps
    assert branch[:2]
    assert cell[:2, :2]
    assert net[:2, :2, :2]

    # test iterability
    for cell in net.cells:
        pass

    for cell in net.cells:
        for branch in cell.branches:
            for comp in branch.comps:
                pass

    for comp in net[0, 0].comps:
        pass


def test_loc_v_comp():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])

    cum_nseg = branch.cumsum_nseg
    nsegs = branch.nseg_per_branch
    branch_ind = 0

    assert np.all(branch.comp(0).show() == branch.loc(0.0).show())
    assert np.all(branch.comp(3).show() == branch.loc(1.0).show())

    inferred_loc = loc_of_index(2, branch_ind, nsegs)
    assert np.all(branch.loc(inferred_loc).show() == branch.comp(2).show())

    inferred_ind = local_index_of_loc(0.4, branch_ind, nsegs)
    assert np.all(branch.comp(inferred_ind).show() == branch.loc(0.4).show())


def test_shape():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(3)], parents=jnp.asarray([-1, 0, 0]))
    net = jx.Network([cell for _ in range(3)])

    assert net.shape == (3, 3 * 3, 3 * 3 * 4)
    assert cell.shape == (3, 3 * 4)
    assert branch.shape == (4,)
    assert comp.shape == ()

    assert net.cell("all").shape == net.shape
    assert cell.branch("all").shape == cell.shape

    assert net.cell("all").shape == (3, 3 * 3, 3 * 3 * 4)
    assert net.cell("all").branch("all").shape == (3, 3 * 3, 3 * 3 * 4)
    assert net.cell("all").branch("all").comp("all").shape == (3, 3 * 3, 3 * 3 * 4)

    assert net.cell(0).shape == (1, 3, 3 * 4)
    assert net.cell(0).branch(0).shape == (1, 1, 4)
    assert net.cell(0).branch(0).comp(0).shape == (1, 1, 1)


def test_set_and_insert():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(5)], parents=jnp.asarray([-1, 0, 0, 1, 1]))
    net = jx.Network([cell for _ in range(5)])
    net1 = deepcopy(net)
    net2 = deepcopy(net)
    net3 = deepcopy(net)
    net4 = deepcopy(net)

    # insert multiple
    net1.cell([0, 1]).branch(0).insert(HH())
    net1.cell(1).branch([0, 1]).insert(HH())
    net1.cell([2, 3]).branch([2, 3]).insert(HH())
    net1.cell(4).branch(4).comp(0).insert(HH())

    net2[[0, 1], 0].insert(HH())
    net2[1, [0, 1]].insert(HH())
    net2[[2, 3], [2, 3]].insert(HH())
    net2[4, 4, 0].insert(HH())

    # set multiple
    net1.cell([0, 1]).branch(0).set("length", 2.0)
    net1.cell(1).branch([0, 1]).set("length", 2.0)
    net1.cell([2, 3]).branch([2, 3]).set("length", 2.0)
    net1.cell(4).branch(4).comp(0).set("length", 2.0)

    net2[[0, 1], 0].set("length", 2.0)
    net2[1, [0, 1]].set("length", 2.0)
    net2[[2, 3], [2, 3]].set("length", 2.0)
    net2[4, 4, 0].set("length", 2.0)

    # insert / set at different levels
    net3.insert(HH())  # insert at net level
    net3.cell(0).insert(HH())  # insert at cell level
    net3.cell(2).branch(2).insert(HH())  # insert at cell level
    net3.cell(4).branch(3).comp(0).insert(HH())  # insert at cell level

    net3.set("length", 2.0)
    net3.cell(0).set("length", 2.0)
    net3.cell(2).branch(2).set("length", 2.0)
    net3.cell(4).branch(3).comp(0).set("length", 2.0)

    net4.insert(HH())
    net4.cell(0).insert(HH())
    net4.cell(2).branch(2).insert(HH())
    net4.cell(4).branch(3).comp(0).insert(HH())

    net4.set("length", 2.0)
    net4.cell(0).set("length", 2.0)
    net4.cell(2).branch(2).set("length", 2.0)
    net4.cell(4).branch(3).comp(0).set("length", 2.0)

    assert all(net1.show() == net2.show())
    assert all(net3.show() == net4.show())

    # insert at into a branch
    branch1 = deepcopy(branch)
    branch2 = deepcopy(branch)

    branch1.comp(0).insert(HH())
    branch2[0].insert(HH())
    assert all(branch1.show() == branch2.show())

    # test insert multiple stimuli
    single_current = jx.step_current(
        i_delay=10.0, i_dur=80.0, i_amp=5.0, delta_t=0.025, t_max=100.0
    )
    batch_of_currents = np.vstack([single_current for _ in range(4)])

    cell1 = deepcopy(cell)
    cell2 = deepcopy(cell)

    cell1.branch(0).stimulate(single_current)
    cell1.branch(1).comp(0).stimulate(single_current)
    cell1.branch(0).stimulate(batch_of_currents)
    cell1.branch(0).record("v")

    cell2[0].stimulate(single_current)
    cell2[1].comp(0).stimulate(single_current)
    cell2[0].stimulate(batch_of_currents)
    cell2.branch(0).record("v")

    assert np.all(cell1.externals["i"] == cell2.externals["i"])
    assert np.all(cell1.external_inds["i"] == cell2.external_inds["i"])
    assert np.all(cell1.recordings == cell2.recordings)


def test_local_indexing():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(5)], parents=jnp.asarray([-1, 0, 0, 1, 1]))
    net = jx.Network([cell for _ in range(2)])

    local_idxs = net.nodes[
        ["local_cell_index", "local_branch_index", "local_comp_index"]
    ]
    idx_cols = ["global_cell_index", "global_branch_index", "global_comp_index"]
    # TODO: Write new and more comprehensive test for local indexing!
    global_index = 0
    for cell_idx in range(2):
        for branch_idx in range(5):
            for comp_idx in range(4):

                # compview = net[cell_idx, branch_idx, comp_idx].show()
                # assert np.all(
                #     compview[idx_cols].values == [cell_idx, branch_idx, comp_idx]
                # )
                assert np.all(
                    local_idxs.iloc[global_index] == [cell_idx, branch_idx, comp_idx]
                )
                global_index += 1


def test_comp_indexing_exception_handling():
    # TODO: Add tests for indexing exceptions
    pass


def test_indexing_a_compartment_of_many_branches():
    comp = jx.Compartment()
    branch1 = jx.Branch(comp, nseg=3)
    branch2 = jx.Branch(comp, nseg=4)
    branch3 = jx.Branch(comp, nseg=5)
    cell1 = jx.Cell([branch1, branch2, branch3], parents=[-1, 0, 0])
    cell2 = jx.Cell([branch3, branch2], parents=[-1, 0])
    net = jx.Network([cell1, cell2])

    # Indexing a single compartment of multiple branches is not supported with `loc`.
    # TODO: Reevaluate what kind of indexing is allowed and which is not!
    # with pytest.raises(NotImplementedError):
    #     net.cell("all").branch("all").loc(0.0)
    # with pytest.raises(NotImplementedError):
    #     net.cell(0).branch("all").loc(0.0)
    # with pytest.raises(NotImplementedError):
    #     net.cell("all").branch(0).loc(0.0)

    # Indexing a single compartment of multiple branches is still supported with `comp`.
    net.cell("all").branch("all").comp(0)
    net.cell(0).branch("all").comp(0)
    net.cell("all").branch(0).comp(0)

    # Indexing many single compartment of multiple branches is always supported.
    net.cell("all").branch("all").loc("all")
    net.cell(0).branch("all").loc("all")
    net.cell("all").branch(0).loc("all")


def test_solve_indexer():
    nsegs = [4, 3, 4, 2, 2, 3, 3]
    cumsum_nseg = cumsum_leading_zero(nsegs)
    idx = JaxleySolveIndexer(cumsum_nseg)
    branch_inds = np.asarray([0, 2])
    assert np.all(idx.first(branch_inds) == np.asarray([0, 7]))
    assert np.all(idx.last(branch_inds) == np.asarray([3, 10]))
    assert np.all(idx.branch(branch_inds) == np.asarray([[0, 1, 2, 3], [7, 8, 9, 10]]))
    assert np.all(idx.lower(branch_inds) == np.asarray([[1, 2, 3], [8, 9, 10]]))
    assert np.all(idx.upper(branch_inds) == np.asarray([[0, 1, 2], [7, 8, 9]]))


# TODO: tests

comp = jx.Compartment()
branch = jx.Branch(comp, nseg=3)
cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
net = jx.Network([cell] * 3)
connect(net[0, 0, 0], net[0, 0, 1], TestSynapse())


# make sure all attrs in module also have a corresponding attr in view
@pytest.mark.parametrize("module", [comp, branch, cell, net])
def test_view_attrs(module):
    # attributes of Module that do not have to exist in View
    exceptions = ["view"]
    # TODO: should be added to View in the future
    exceptions += [
        "cumsum_nseg",
        "_internal_node_inds",
        "par_inds",
        "child_inds",
        "child_belongs_to_branchpoint",
        "solve_indexer",
        "_comp_edges",
        "_n_nodes",
        "_data_inds",
        "_indices_jax_spsolve",
        "_indptr_jax_spsolve",
    ]  # for base/comp
    exceptions += ["comb_children"]  # for cell
    exceptions += [
        "cells_list",
        "cumsum_nbranchpoints_per_cell",
        "_cumsum_nseg_per_cell",
    ]  # for network
    exceptions += ["cumsum_nbranches"]  # HOTFIX #TODO: take care of this

    for name, attr in module.__dict__.items():
        if name not in exceptions:
            # check if attr is in view
            view = View(module)
            assert hasattr(view, name), f"View missing attribute: {name}"
            # check if types match
            assert type(getattr(module, name)) == type(
                getattr(view, name)
            ), f"Type mismatch: {name}, Module type: {type(getattr(module, name))}, View type: {type(getattr(view, name))}"


# TODO: test filter for modules and check for param sharing
# add test local_indexing and global_indexing
# add cell.comp (branch is skipped also for param sharing)
# add tests for new features i.e. iter, context, scope
