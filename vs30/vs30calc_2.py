import time
from functools import partial
import logging
import os
from shutil import copyfile, rmtree
import sys
import tempfile
from multiprocessing import get_context

import numpy as np
import pandas as pd
from pyproj import Transformer

from vs30 import (
    model_fixed_weightedaverage,
    model_geology_new2,
    model_terrain_2,
    mvn,
    params,
    sites_cluster,
    sites_load_NSHM2022,
)

WGS2NZTM = Transformer.from_crs(4326, 2193, always_xy=True)

# ---------------------------------------------------------------------
# Globals for multiprocessing workers
# ---------------------------------------------------------------------
_GLOBAL_TABLE = None
_GLOBAL_SITES = None
_GLOBAL_MODEL_NAME = None


def init_mvn_worker(table, sites, model_name):
    """
    Initializer for each worker process.
    Large shared inputs are set once per worker instead of being passed
    repeatedly in every task.
    """
    global _GLOBAL_TABLE, _GLOBAL_SITES, _GLOBAL_MODEL_NAME
    _GLOBAL_TABLE = table
    _GLOBAL_SITES = sites
    _GLOBAL_MODEL_NAME = model_name


def make_index_chunks(n_rows: int, chunk_size: int = 2000):
    """
    Split row indices into chunks like [(0, 2000), (2000, 4000), ...]
    """
    return [(i, min(i + chunk_size, n_rows)) for i in range(0, n_rows, chunk_size)]


def mvn_worker_index_range(idx_range):
    """
    Worker function:
    - slices the global table by row range
    - computes mvn_table for that chunk
    - writes result to temporary .npy file
    - returns only small metadata (start, end, temp path)

    This avoids sending large numpy arrays back through multiprocessing pipes.
    """
    start, end = idx_range

    chunk = _GLOBAL_TABLE.iloc[start:end].copy()

    arr = mvn.mvn_table(
        chunk,
        sites=_GLOBAL_SITES,
        model_name=_GLOBAL_MODEL_NAME,
    )

    fd, tmp_path = tempfile.mkstemp(suffix=".npy", prefix="mvn_chunk_")
    os.close(fd)
    np.save(tmp_path, arr)

    return (start, end, tmp_path)


def run_parallel_mvn_table(
    table: pd.DataFrame,
    measured_sites: pd.DataFrame,
    model_name: str,
    n_procs: int,
    chunk_size: int = 2000,
):
    """
    Parallel MVN computation without passing large arrays through pipes.

    Returns
    -------
    np.ndarray
        Concatenated result array from mvn.mvn_table over all chunks.
    """
    chunks = make_index_chunks(len(table), chunk_size=chunk_size)
    results = []

    # spawn is safer than fork for heavy scientific workloads
    ctx = get_context("spawn")

    with ctx.Pool(
        processes=n_procs,
        initializer=init_mvn_worker,
        initargs=(table, measured_sites, model_name),
        maxtasksperchild=20,
    ) as pool:
        for item in pool.imap_unordered(mvn_worker_index_range, chunks):
            results.append(item)

    # restore original row order
    results.sort(key=lambda x: x[0])

    arrays = []
    try:
        for _, _, tmp_path in results:
            arrays.append(np.load(tmp_path))
        out = np.concatenate(arrays, axis=0)
    finally:
        for _, _, tmp_path in results:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return out


def run_vs30calc(
    p_paths: params.PathsParams,
    p_sites: params.SitesParams,
    p_grid: params.GridParams | None,
    p_ll: params.LLFileParams | None,
    p_geol: params.GeologyParams,
    p_terr: params.TerrainParams,
    p_comb: params.CombinationParams,
    n_procs: int,
):
    """
    Run the Vs30 calculation for the specified parameters.

    Parameters
    ----------
    p_paths : params.PathsParams
        Paths parameters.
    p_sites : params.SitesParams
        Sites parameters.
    p_grid : params.GridParams | None
        Grid parameters.
    p_ll : params.LLFileParams | None
        Latitude/longitude file parameters.
    p_geol : params.GeologyParams
        Geology model parameters.
    p_terr : params.TerrainParams
        Terrain model parameters.
    p_comb : params.CombinationParams
        Combined model parameters.
    n_procs : int
        Number of processes to use.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running Vs30Calc with {n_procs} processes")

    # -----------------------------------------------------------------
    # Output setup
    # -----------------------------------------------------------------
    if os.path.exists(p_paths.out):
        if p_paths.overwrite:
            rmtree(p_paths.out)
        else:
            sys.exit("output exists")
    os.makedirs(p_paths.out)

    # -----------------------------------------------------------------
    # Input locations
    # -----------------------------------------------------------------
    if p_ll is not None:
        logger.info("loading locations...")
        table = pd.read_csv(
            p_ll.ll_path,
            usecols=(p_ll.lon_col_ix, p_ll.lat_col_ix),
            names=["longitude", "latitude"],
            engine="c",
            skiprows=p_ll.skip_rows,
            dtype=np.float64,
            sep=p_ll.col_sep,
        )
        table["easting"], table["northing"] = WGS2NZTM.transform(
            table.longitude.values, table.latitude.values
        )
        table_points = table[["easting", "northing"]].values

    # -----------------------------------------------------------------
    # Measured sites
    # -----------------------------------------------------------------
    logger.info("loading measured sites...")
    measured_sites = sites_load_NSHM2022.load_vs(source=p_sites.source)
    measured_sites_points = np.column_stack(
        (measured_sites.easting.values, measured_sites.northing.values)
    )

    MODEL_MAPPING = {
        "geology": model_geology_new2,
        "terrain": model_terrain_2,
    }

    # -----------------------------------------------------------------
    # Model loop
    # -----------------------------------------------------------------
    tiffs = []
    tiffs_mvn = []

    for model_setup in [p_geol, p_terr]:
        if model_setup.update == "off":
            continue

        logger.info(f"{model_setup.name} model...")
        model_module = MODEL_MAPPING[model_setup.name]

        # -------------------------------------------------------------
        # Model measured update
        # -------------------------------------------------------------
        logger.info("    model measured update...")
        start_time = time.time()

        measured_sites[f"{model_setup.letter}id"] = model_module.model_id(
            measured_sites_points
        )

        if model_setup.update == "prior":
            model_table = model_module.model_prior()

        elif model_setup.update == "posterior_paper":
            model_table = model_module.model_posterior_paper()

        elif model_setup.update == "posterior":
            if p_sites.source == "cpt":
                measured_sites = sites_cluster.cluster(...)
                model_table = model_module.model_prior()
                model_table = model_fixed_weightedaverage.cluster_update(
                            model_table, measured_sites, model_setup.letter
                )
            else:
                model_table = model_fixed_weightedaverage.posterior(
                    measured_sites, 
                    f"{model_setup.letter}id",
                    method="weighted"  # 기본값
                    )

        logger.info(f"    took: {time.time() - start_time:.2f}s")

        # -------------------------------------------------------------
        # Model at measured sites
        # -------------------------------------------------------------
        logger.info("    model at measured sites...")
        start_time = time.time()

        (
            measured_sites[f"{model_setup.name}_vs30"],
            measured_sites[f"{model_setup.name}_stdv"],
        ) = model_module.model_val(
            measured_sites[f"{model_setup.letter}id"].values,
            model_table,
            model_setup,
            paths=p_paths,
            points=measured_sites_points,
            grid=p_grid,
        ).T

        logger.info(f"    took: {time.time() - start_time:.2f}s")

        # -------------------------------------------------------------
        # Point mode
        # -------------------------------------------------------------
        if p_ll is not None:
            logger.info("    model points...")
            start = time.time()

            table[f"{model_setup.letter}id"] = model_module.model_id(table_points)

            logger.info(f"    took: {time.time() - start:.2f}s")

            (
                table[f"{model_setup.name}_vs30"],
                table[f"{model_setup.name}_stdv"],
            ) = model_module.model_val(
                table[f"{model_setup.letter}id"].values,
                model_table,
                model_setup,
                paths=p_paths,
                points=table_points,
                grid=p_grid,
            ).T

            # ---------------------------------------------------------
            # MVN update at points
            # ---------------------------------------------------------
            logger.info("    measured mvn...")
            start_time = time.time()

            if n_procs == 1:
                (
                    table[f"{model_setup.name}_mvn_vs30"],
                    table[f"{model_setup.name}_mvn_stdv"],
                ) = mvn.mvn_table(
                    table,
                    sites=measured_sites,
                    model_name=model_setup.name,
                ).T
            else:
                arr = run_parallel_mvn_table(
                    table=table,
                    measured_sites=measured_sites,
                    model_name=model_setup.name,
                    n_procs=n_procs,
                    chunk_size=2000,   # reduce to 1000 if still memory heavy
                )
                (
                    table[f"{model_setup.name}_mvn_vs30"],
                    table[f"{model_setup.name}_mvn_stdv"],
                ) = arr.T

            logger.info(f"    took: {time.time() - start_time:.2f}s")

        # -------------------------------------------------------------
        # Raster mode
        # -------------------------------------------------------------
        else:
            logger.info("    model map...")
            tiffs.append(
                model_module.model_val_map(p_paths, p_grid, model_table, model_setup)
            )

            logger.info("    measured mvn...")
            tiffs_mvn.append(
                mvn.mvn_tiff(p_paths.out, model_setup.name, measured_sites, n_procs)
            )

    # -----------------------------------------------------------------
    # Combine geology and terrain
    # -----------------------------------------------------------------
    if p_geol.update != "off" and p_terr.update != "off":
        logger.info("combining geology and terrain...")
        start_time = time.time()

        if p_ll is not None:
            for prefix in ["", "mvn_"]:
                table[f"{prefix}vs30"], table[f"{prefix}stdv"] = (
                    model_fixed_weightedaverage.combine_models(
                        p_comb,
                        table[f"geology_{prefix}vs30"],
                        table[f"geology_{prefix}stdv"],
                        table[f"terrain_{prefix}vs30"],
                        table[f"terrain_{prefix}stdv"],
                    )
                )
        else:
            model_fixed_weightedaverage.combine_tiff(
                p_paths.out, "combined.tif", p_grid, p_comb, *tiffs
            )
            model_fixed_weightedaverage.combine_tiff(
                p_paths.out, "combined_mvn.tif", p_grid, p_comb, *tiffs_mvn
            )

        logger.info(f"{time.time() - start_time:.2f}s")

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------
    measured_sites.to_csv(
        os.path.join(p_paths.out, "measured_sites.csv"),
        na_rep="NA",
        index=False,
    )

    if p_ll is not None:
        table.to_csv(
            os.path.join(p_paths.out, "vs30points.csv"),
            na_rep="NA",
            index=False,
        )
        copyfile(
            os.path.join(sites_load_NSHM2022.data, "qgis_points.qgz"),
            os.path.join(p_paths.out, "qgis.qgz"),
        )
    else:
        copyfile(
            os.path.join(sites_load_NSHM2022.data, "qgis_rasters.qgz"),
            os.path.join(p_paths.out, "qgis.qgz"),
        )

    logger.info("complete.")


if __name__ == "__main__":
    # Example:
    # p_paths, p_sites, p_grid, p_ll, p_geol, p_terr, p_comb should be
    # prepared elsewhere exactly as in your original workflow.
    #
    # run_vs30calc(
    #     p_paths=p_paths,
    #     p_sites=p_sites,
    #     p_grid=p_grid,
    #     p_ll=p_ll,
    #     p_geol=p_geol,
    #     p_terr=p_terr,
    #     p_comb=p_comb,
    #     n_procs=4,
    # )
    pass
