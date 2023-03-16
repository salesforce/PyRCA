import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from pyrca.thirdparty.causallearn.utils.cit import chisq
from pyrca.thirdparty.causallearn.utils.cit import CIT
from pyrca.thirdparty.rcd.utils import SkeletonDiscovery

import time


def run_multi_phase(normal_df, anomalous_df, config):
    f_child_union = normal_df.columns
    mi_union = []
    i = 0
    prev = len(f_child_union)

    # Phase-1
    while True:
        start = time.time()
        f_child_union, mi, ci_tests = __run_level(normal_df.loc[:, f_child_union],
                                                anomalous_df.loc[:, f_child_union],
                                                config
                                                )
        if config['verbose']:
            print(f"Level-{i}: variables {len(f_child_union)} | time {time.time() - start}")
        i += 1
        mi_union += mi
        # Phase-1 with only one level

        len_child = len(f_child_union)
        # If found gamma nodes or if running the current level did not remove any node
        if len_child <= config['gamma'] or len_child == prev: break
        prev = len(f_child_union)

    # Phase-2
    mi_union = []
    new_nodes = f_child_union
    rc, _, mi, ci = __top_k_rc(normal_df.loc[:, new_nodes],
                                    anomalous_df.loc[:, new_nodes],
                                    mi=mi_union,
                                    config=config,
                                    )
    ci_tests += ci

    return rc, ci_tests


def __run_level(normal_df, anomalous_df, config):
    ci_tests = 0
    chunks = __create_chunks(normal_df, config)
    if config['verbose']:
        print(f"Created {len(chunks)} subsets")

    f_child_union = list()
    mi_union = list()
    f_child = list()
    for c in chunks:
        # Try this segment with multiple values of alpha until we find at least one node
        rc, _, mi, ci = __top_k_rc(normal_df.loc[:, c],
                                        anomalous_df.loc[:, c],
                                        min_nodes=1,
                                        config=config,
                                        )
        f_child_union += rc
        mi_union += mi
        ci_tests += ci
        if config['verbose']:
            f_child.append(rc)

    if config['verbose']:
        print(f"Output of individual chunk {f_child}")
        print(f"Total nodes in mi => {len(mi_union)} | {mi_union}")

    return f_child_union, mi_union, ci_tests


# Split the dataset into multiple subsets
def __create_chunks(df, config):
    chunks = list()
    names = np.random.permutation(df.columns)
    for i in range(df.shape[1] // config['gamma'] + 1):
        chunks.append(names[i * config['gamma']:(i * config['gamma']) + config['gamma']])

    if len(chunks[-1]) == 0:
        chunks.pop()
    return chunks


# Equivelant to \Psi-PC from the main paper
def __top_k_rc(normal_df, anomalous_df, mi=[],
               min_nodes=-1, config=None,
               ):
    data = __preprocess_for_fnode(normal_df, anomalous_df, config)

    if min_nodes == -1:
        # Order all nodes (if possible) except F-node
        min_nodes = len(data.columns) - 1
    assert (min_nodes < len(data))

    G = None
    no_ci = 0
    i_to_labels = {i: name for i, name in enumerate(data.columns)}
    labels_to_i = {name: i for i, name in enumerate(data.columns)}

    _preprocess_mi = lambda l: [labels_to_i.get(i) for i in l]
    _postprocess_mi = lambda l: [i_to_labels.get(i) for i in list(filter(None, l))]
    processed_mi = _preprocess_mi(mi)
    _run_pc = lambda alpha: __run_pc(data, alpha, mi=processed_mi,
                                          labels=i_to_labels, config=config)

    rc = []
    for i in np.arange(config['start_alpha'], config['alpha_limit'], config['alpha_step']):
        cg = _run_pc(i)
        G = cg.nx_graph
        no_ci += cg.no_ci_tests

        if G is None: continue

        f_neigh = __get_fnode_child(G, config)
        new_neigh = [x for x in f_neigh if x not in rc]
        if len(new_neigh) == 0:
            continue
        else:
            f_p_values = cg.p_values[-1][[labels_to_i.get(key) for key in new_neigh]]
            rc += __order_neighbors(new_neigh, f_p_values)

        if len(rc) == min_nodes: break

    return (rc, G, _postprocess_mi(cg.mi), no_ci)


def __preprocess_for_fnode(normal_df, anomalous_df, config):
    df = __add_fnode(normal_df, anomalous_df, config)
    if df is None: return None

    return __discretize(df, config["bins"], config) if config["bins"] is not None else df


def __add_fnode(normal_df, anomalous_df, config):
    normal_df[config['f_node']] = '0'
    anomalous_df[config['f_node']] = '1'
    return pd.concat([normal_df, anomalous_df])


# Run PC (only the skeleton phase) on the given dataset.
# The last column of the data *must* be the F-node
def __run_pc(data, alpha, labels={}, mi=[], config=None):
    if labels == {}:
        labels = {i: name for i, name in enumerate(data.columns)}

    np_data = data.to_numpy()
    indep_test = CIT(data, config['ci_test'])
    if config['localized']:
        f_node = np_data.shape[1] - 1
        # Localized PC
        cg = SkeletonDiscovery.local_skeleton_discovery(np_data, f_node, alpha,
                                                        indep_test=indep_test, mi=mi,
                                                        labels=labels, verbose=config['verbose'])
    else:
        cg = SkeletonDiscovery.skeleton_discovery(np_data, alpha, indep_test=indep_test,
                                                  background_knowledge=None,
                                                  stable=False, verbose=config['verbose'],
                                                  labels=labels, show_progress=False)

    cg.to_nx_graph()
    return cg


def __get_fnode_child(G, config):
    return [*G.successors(config['f_node'])]


def __discretize(data, bins, config):
    d = data.iloc[:, :-1]
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    discretizer.fit(d)
    disc_d = discretizer.transform(d)
    disc_d = pd.DataFrame(disc_d, columns=d.columns.values.tolist())
    disc_d[config['f_node']] = data[config['f_node']].tolist()

    for c in disc_d:
        disc_d[c] = disc_d[c].astype(int)

    return disc_d


def __order_neighbors(neigh, p_values):
    _neigh = neigh.copy()
    _p_values = p_values.copy()
    stack = []

    while len(_neigh) != 0:
        i = np.argmax(_p_values)
        node = _neigh[i]
        stack = [node] + stack
        _neigh.remove(node)
        _p_values = np.delete(_p_values, i)
    return stack