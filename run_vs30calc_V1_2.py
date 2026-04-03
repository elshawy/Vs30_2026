#!/usr/bin/env python
"""
Calculate Vs30 over a region (default) or specify points at which to output for instead.
"""
from pyproj import Transformer

from vs30 import (
    params_3,
    vs30calc_V1_2,
)

WGS2NZTM = Transformer.from_crs(4326, 2193, always_xy=True)

# work on ~50 points per process
PROCESS_CHUNK_SIZE = 50

def main():
    p_paths, p_sites, p_grid, p_ll, p_geol, p_terr, p_comb, nproc = params_3.load_args()
    vs30calc_V1_2.run_vs30calc(
        p_paths,
        p_sites,
        p_grid,
        p_ll,
        p_geol,
        p_terr,
        p_comb,
        nproc,
    )

if __name__ == "__main__":
    main()
