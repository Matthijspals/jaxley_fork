import numpy as np


def read_swc(fname):
    """Read an SWC file and bring morphology into `neurax` compatible formats."""
    content = np.loadtxt(fname)

    branches = _split_into_branches(content)
    branches = _remove_single_branch_artifacts(branches)
    # print("branches", branches)

    first_val = np.asarray([b[0] for b in branches])
    # print("first_val", first_val)
    sorting = np.argsort(first_val, kind="mergesort")
    # print("sorting", sorting)
    sorted_branches = [branches[s] for s in sorting]

    parents = _build_parents(sorted_branches)
    pathlengths = _compute_pathlengths(sorted_branches, content[:, 2:5])
    endpoint_radiuses = _extract_endpoint_radiuses(sorted_branches, content[:, 5])
    start_radius = content[0, 5]
    return parents, pathlengths, endpoint_radiuses, start_radius


def _remove_single_branch_artifacts(branches):
    """Check that all parents have two children. No only childs allowed!

    See GH #32. The reason this happens is that some branches (without branchings)
    are interrupted in their tracing. Here, we fuse these interrupted branches.
    """
    first_val = np.asarray([b[0] for b in branches])
    vals, counts = np.unique(first_val[1:], return_counts=True)
    one_vals = vals[counts == 1]
    for one_val in one_vals:
        loc = np.where(first_val == one_val)[0][0]
        solo_branch = branches[loc]
        del branches[loc]
        new_branches = []
        for b in branches:
            if b[-1] == one_val:
                new_branches.append(b + solo_branch)
            else:
                new_branches.append(b)
        branches = new_branches

    return branches


def _split_into_branches(content):
    prev_ind = None
    n_branches = 0
    branch_inds = []
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        if current_parent != prev_ind:
            branch_inds.append(int(current_parent))
            n_branches += 1
        prev_ind = current_ind

    all_branches = []
    current_branch = []
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        if current_parent in branch_inds[1:]:
            all_branches.append(current_branch)
            current_branch = [int(current_parent), int(current_ind)]
        else:
            current_branch.append(int(current_ind))
    all_branches.append(current_branch)

    return all_branches


def _build_parents(all_branches):
    parents = [None] * len(all_branches)
    all_last_inds = [b[-1] for b in all_branches]
    for i, b in enumerate(all_branches):
        parent_ind = b[0]
        ind = np.where(np.asarray(all_last_inds) == parent_ind)[0]
        if len(ind) > 0 and ind != i:
            parents[i] = ind[0]
        else:
            parents[i] = -1

    return parents


def _extract_endpoint_radiuses(all_branches, radiuses):
    endpoint_radiuses = []
    for b in all_branches:
        branch_endpoint = b[-1]
        # Beause SWC starts counting at 1, but numpy counts from 0.
        ind_of_branch_endpoint = branch_endpoint - 1
        endpoint_radiuses.append(radiuses[ind_of_branch_endpoint])
    return endpoint_radiuses


def _compute_pathlengths(all_branches, coords):
    branch_pathlengths = []
    for b in all_branches:
        coords_in_branch = coords[np.asarray(b) - 1]
        point_diffs = np.diff(coords_in_branch, axis=0)
        dists = np.sqrt(
            point_diffs[:, 0] ** 2 + point_diffs[:, 1] ** 2 + point_diffs[:, 2] ** 2
        )
        branch_pathlengths.append(np.sum(dists))
    return branch_pathlengths
