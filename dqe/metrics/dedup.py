"""Cross-episode near-duplicate detection.

Builds a fixed-length, z-normalised fingerprint per episode and finds pairs
whose normalised RMS distance falls below a threshold (optionally only within
the same task). Within each duplicate cluster the longest episode is kept and
the rest are flagged, mirroring RLDX's redundancy pruning of its demo pool.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def find_duplicates(fingerprints: Dict[int, np.ndarray],
                    lengths: Dict[int, int],
                    task_of: Dict[int, int],
                    near_dup_dist: float,
                    within_task_only: bool
                    ) -> Tuple[Dict[int, int], List[Tuple[int, int, float]]]:
    """Return ({episode -> representative kept episode}, [(dup, rep, dist)]).

    An episode mapped to itself is unique/kept. Distance is RMS over the
    fingerprint, divided by sqrt(dim) so it is comparable across datasets.
    """
    eps = sorted(fingerprints)
    if not eps:
        return {}, []
    mat = np.stack([fingerprints[e] for e in eps])
    dim = mat.shape[1]
    tasks = np.array([task_of.get(e, -1) for e in eps])

    parent = {e: e for e in eps}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    dup_pairs: List[Tuple[int, int, float]] = []
    n = len(eps)
    for i in range(n):
        di = mat[i]
        # vectorised distance from i to all j>i
        diff = mat[i + 1:] - di
        if diff.shape[0] == 0:
            continue
        dists = np.sqrt((diff ** 2).mean(axis=1))
        for off, dist in enumerate(dists):
            j = i + 1 + off
            if within_task_only and tasks[i] != tasks[j]:
                continue
            if dist <= near_dup_dist:
                ri, rj = find(eps[i]), find(eps[j])
                if ri != rj:
                    # keep the longer episode as representative
                    keep, drop = (ri, rj) if lengths.get(ri, 0) >= lengths.get(rj, 0) else (rj, ri)
                    parent[drop] = keep
                dup_pairs.append((eps[j], eps[i], float(dist)))

    rep = {e: find(e) for e in eps}
    return rep, dup_pairs
