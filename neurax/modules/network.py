from typing import Dict, List, Optional, Callable
import itertools

import numpy as np
import jax.numpy as jnp
import pandas as pd

from neurax.modules.base import Module, View
from neurax.modules.cell import Cell, CellView
from neurax.connection import Connection
from neurax.network import Connectivity
from neurax.integrate import _step_synapse
from neurax.stimulus import get_external_input
from neurax.cell import merge_cells, _compute_index_of_kid, cum_indizes_of_kids


class Network(Module):
    network_params: Dict = {}
    network_states: Dict = {}

    def __init__(self, cells: List[Cell], conns: List[List[Connection]]):
        """Initialize network of cells and synapses.

        Args:
            cells (List[Cell]): _description_
            conns (List[List[Connection]]): _description_
        """
        super().__init__()
        self._init_params_and_state(self.network_params, self.network_states)
        self._append_to_params_and_state(cells)

        self.cells = cells
        self.nseg = cells[0].nseg

        assert isinstance(conns, list), "conns must be a list."
        for conn in conns:
            assert isinstance(conn, list), "conns must be a list of lists."
        self.connectivities = [Connectivity(conn, self.nseg) for conn in conns]

        # Define morphology of synapses.
        self.pre_syn_inds = [c.pre_syn_inds for c in self.connectivities]
        self.pre_syn_cell_inds = [c.pre_syn_cell_inds for c in self.connectivities]
        self.grouped_post_syn_inds = [
            c.grouped_post_syn_inds for c in self.connectivities
        ]
        self.grouped_post_syns = [c.grouped_post_syns for c in self.connectivities]

        self.initialized_morph = False
        self.initialized_conds = True

    def __getattr__(self, key):
        assert key == "cell"
        return CellView(self, self.nodes)

    def init_morph(self):
        self.nbranches = [cell.nbranches for cell in self.cells]
        self.cumsum_num_branches = jnp.cumsum(jnp.asarray([0] + self.nbranches))
        self.max_num_kids = 4
        # for c in self.cells:
        #     assert (
        #         self.max_num_kids == c.max_num_kids
        #     ), "Different max_num_kids between cells."

        parents = [cell.parents for cell in self.cells]
        self.comb_parents = jnp.concatenate(
            [p.at[1:].add(self.cumsum_num_branches[i]) for i, p in enumerate(parents)]
        )
        self.comb_parents_in_each_level = merge_cells(
            self.cumsum_num_branches,
            [cell.parents_in_each_level for cell in self.cells],
        )
        self.comb_branches_in_each_level = merge_cells(
            self.cumsum_num_branches,
            [cell.branches_in_each_level for cell in self.cells],
            exclude_first=False,
        )

        # Prepare indizes for solve
        comb_ind_of_kids = jnp.concatenate(
            [jnp.asarray(_compute_index_of_kid(cell.parents)) for cell in self.cells]
        )
        self.comb_cum_kid_inds = cum_indizes_of_kids(
            [comb_ind_of_kids], self.max_num_kids, reset_at=[-1, 0]
        )[0]
        comb_ind_of_kids_in_each_level = [
            comb_ind_of_kids[bil] for bil in self.comb_branches_in_each_level
        ]
        self.comb_cum_kid_inds_in_each_level = cum_indizes_of_kids(
            comb_ind_of_kids_in_each_level, self.max_num_kids, reset_at=[0]
        )

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(
                comp_index=np.arange(self.nseg * sum(self.nbranches)).tolist(),
                branch_index=(
                    np.arange(self.nseg * sum(self.nbranches)) // self.nseg
                ).tolist(),
                cell_index=list(
                    itertools.chain(
                        *[[i] * (self.nseg * b) for i, b in enumerate(self.nbranches)]
                    )
                ),
            )
        )

        self.initialized_morph = True

    def init_conds(self):
        # Initially, the coupling conductances are set to `None`. They have to be set
        # by calling `.set_axial_resistivities()`.
        self.coupling_conds_fwd = [c.coupling_conds_fwd for c in self.cells]
        self.coupling_conds_bwd = [c.coupling_conds_bwd for c in self.cells]
        self.branch_conds_fwd = [c.branch_conds_fwd for c in self.cells]
        self.branch_conds_bwd = [c.branch_conds_bwd for c in self.cells]
        self.summed_coupling_conds = [c.summed_coupling_conds for c in self.cells]
        self.initialized_conds = True

    def step(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        i_inds: jnp.ndarray,
        i_current: jnp.ndarray,
    ):
        """Step for a single compartment.

        Args:
            u: The full state vector, including states of all channels and the voltage.
            dt: Time step.

        Returns:
            Next state. Same shape as `u`.
        """
        nbranches = self.nbranches
        nseg_per_branch = self.nseg

        if self.branch_conds_fwd is None:
            self.init_branch_conds()

        voltages = u["voltages"]

        # Parameters have to go in here.
        new_channel_states, (v_terms, const_terms) = self.step_channels(
            u, dt, self.branches[0].compartments[0].channels, self.params
        )

        # External input.
        i_ext = get_external_input(
            voltages, i_inds, i_current, self.params["radius"], self.params["length"]
        )

        # Step of the synapse.
        new_syn_states, syn_voltage_terms, syn_constant_terms = _step_synapse(
            u,
            self.syn_channels,
            self.params,
            dt,
            self.cumsum_num_branches,
            self.pre_syn_cell_inds,
            self.pre_syn_inds,
            self.grouped_post_syn_inds,
            self.grouped_post_syns,
            self.nseg,
        )

        new_voltages = self.step_voltages(
            voltages=jnp.reshape(voltages, (nbranches, nseg_per_branch)),
            voltage_terms=jnp.reshape(v_terms, (nbranches, nseg_per_branch)),
            constant_terms=jnp.reshape(const_terms, (nbranches, nseg_per_branch))
            + jnp.reshape(i_ext, (nbranches, nseg_per_branch)),
            coupling_conds_bwd=self.coupling_conds_bwd,
            coupling_conds_fwd=self.coupling_conds_fwd,
            summed_coupling_conds=self.summed_coupling_conds,
            branch_cond_fwd=self.branch_conds_fwd,
            branch_cond_bwd=self.branch_conds_bwd,
            num_branches=self.nbranches,
            parents=self.parents,
            kid_inds_in_each_level=self.kid_inds_in_each_level,
            max_num_kids=4,
            parents_in_each_level=self.parents_in_each_level,
            branches_in_each_level=self.branches_in_each_level,
            tridiag_solver="thomas",
            delta_t=dt,
        )
        final_state = new_channel_states[0]
        final_state["voltages"] = new_voltages.flatten(order="C")
        return final_state
