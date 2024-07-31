import datetime
import logging
import os
import pickle
import time
from typing import Callable

import numpy as np
import numpy.typing as npt

from umni_crl import umni_crl
import utils

scm_type = "linear"

if scm_type == "linear":
    from scm.linear import LinearSCM as SCM
else:
    raise ValueError(f"{scm_type=} is not recognized!")

ATOL_EIGV = 5e-2
ATOL_CI_TEST = 5e-2

ATOL_EFF_NZ = 1e-1
DECODER_MIN_COND_NUM = 1e-1

d_mat_choice_list = ["SN", "MN-uppertri"]
d_mat_choice = "MN-uppertri"

if __name__ == "__main__":
    nd_list = [
        (5, 5),
    ]

    fill_rate = 0.5
    nsamples = 100_000
    nruns = 100
    np_rng = np.random.default_rng()

    # Score computation/estimation settings
    estimate_score_fns = True
    nsamples_for_se = nsamples
    enable_gaussian_score_est = True

    # SCM settings
    hard_intervention = True
    hard_graph_postprocess = True
    type_int = "hard int"
    var_change_mech = "scale"
    var_change = 0.25

    # no need to randomize top_order in UMN setting. Just randomize D columns if sampling wrt some constraints, e.g., uppertri
    randomize_top_order = False

    # Result dir setup
    run_name = (
        scm_type
        + "_"
        + ("hard" if hard_intervention else "soft")
        + "_"
        + f"ns{nsamples_for_se/1000:g}k"
        + "_"
        + f"nr{nruns}"
        + "_"
        + (
            "gt"
            if not estimate_score_fns
            else (
                "ss"
                if not enable_gaussian_score_est or scm_type != "linear"
                else "gaus"
            )
        )
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    run_dir = os.path.join("results", "umni_crl", run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Logger setup
    log_file = os.path.join(run_dir, "out.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)-12s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    log_file_h = logging.FileHandler(log_file)
    log_file_h.setFormatter(log_formatter)
    log_file_h.setLevel(logging.DEBUG)
    log_console_h = logging.StreamHandler()
    log_console_h.setFormatter(log_formatter)
    log_console_h.setLevel(logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.addHandler(log_file_h)
    logger.addHandler(log_console_h)

    logger.info(f"Logging to {log_file}")

    config = {
        "ATOL_EIGV": ATOL_EIGV,
        "ATOL_CI_TEST": ATOL_CI_TEST,
        "scm_type": scm_type,
        "hard_intervention": hard_intervention,
        "hard_graph_postprocess": hard_graph_postprocess,
        "var_change_mech": var_change_mech,
        "var_change": var_change,
        "nd_list": nd_list,
        "nruns" : nruns,
        "nsamples": nsamples,
        "estimate_score_fns": estimate_score_fns,
        "nsamples_for_se": nsamples_for_se,
    }
    with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    results = {
        (n, d): [
            {
                # Inputs
                #"scm": SCM.__new__(SCM),
                "intervention_order": np.empty(n, dtype=np.int_),
                "decoder": np.empty((d, n)),
                "encoder": np.empty((n, d)),
                # Outputs
                "is_run_ok": False,
                # `_obtain_top_order`
                "top_order": np.empty((n,), dtype=np.int_),
                # `_minimize_score_variations`
                "hat_g_s": np.empty((n, n), dtype=bool),
                "hat_enc_s": np.empty((n, d)),
                # Analysis
                "shd_s": 0.0,
                "edge_precision_s": 0.0,
                "edge_recall_s": 0.0,
                "norm_z_err_s": 0.0,
                "extra_nz_in_eff_s": 0,
                "distance_to_pi": 0,
                # `_unmixing_procedure`
                "hat_g_h": np.empty((n, n), dtype=bool),
                "hat_enc_h": np.empty((n, d)),
                "w_mat_s": np.empty((n, n)),
                # Analysis
                "shd_h": 0.0,
                "edge_precision_h": 0.0,
                "edge_recall_h": 0.0,
                "norm_z_err_h": 0.0,
                "extra_nz_in_eff_h": 0,
            }
            for _ in range(nruns)
        ]
        for (n, d) in nd_list
    }

    t0 = time.time()

    for nd_idx, (n, d) in enumerate(nd_list):
        logger.info(f"Starting {(n, d) = }")

        for run_idx in range(nruns):
            if run_idx % 10 == 10 - 1:
                logger.info(f"{(n, d) = }, {run_idx = }")

            results_run = results[n, d][run_idx]

            # Build the decoder in two steps:
            # 1: Uniformly random selection of column subspace
            # TODO: Theoretically ensure this is indeed uniform
            import scipy.stats  # type: ignore
            decoder_q: npt.NDArray[np.float_] = scipy.stats.ortho_group(d, np_rng).rvs()[:, :n]  # type: ignore

            # 2: Random mixing within the subspace
            decoder_r = np_rng.random((n, n)) - 0.5
            decoder_r_svs = np.linalg.svd(decoder_r, compute_uv=False)
            while decoder_r_svs[-1] / decoder_r_svs[0] < DECODER_MIN_COND_NUM:
                decoder_r = np_rng.random((n, n)) - 0.5
                decoder_r_svs = np.linalg.svd(decoder_r, compute_uv=False)

            # Then, the full decoder is the composition of these transforms
            decoder = decoder_q @ decoder_r
            encoder = np.linalg.pinv(decoder)

            scm = SCM(
                n,
                fill_rate,
                randomize_top_order=randomize_top_order,
                np_rng=np_rng,
            )

            d_rand_column = np_rng.permutation(n)
            if d_mat_choice == "SN":
                # D matrix sampling - DEBUG with SN interventions
                d_mat = np.eye(n, dtype=bool)
                d_mat = d_mat[:,d_rand_column]
            elif d_mat_choice == "MN-uppertri":
                # D matrix sampling - alternative 1: uppertri
                d_mat = np.eye(n, dtype=bool) | np.triu(np_rng.random((n, n)) <= fill_rate)
                while np.linalg.det(d_mat) == 0:
                    d_mat = np.eye(n, dtype=bool) | np.triu(np_rng.random((n, n)) <= fill_rate)
                d_mat = d_mat[:,d_rand_column]
            else:
                raise ValueError(f"{d_mat_choice=} is not recognized!")


            envs = [list[int]()] + [[i for i in range(n) if d_mat[i, j]] for j in range(n)]

            z_samples = np.stack(
                [
                    scm.sample(
                        (nsamples,),
                        nodes_int=env,
                        type_int=type_int,
                        var_change_mech=var_change_mech,
                        var_change=var_change,
                    )
                    for env in envs
                ]
            )
            z_samples_norm = (z_samples.__pow__(2).sum() ** (0.5))

            x_samples = decoder @ z_samples

            # Evaluate score functions on the same data points
            if estimate_score_fns:
                # Use estimated (noisy) score functions

                z_samples_for_se = np.stack(
                    [
                        scm.sample(
                            (nsamples_for_se,),
                            nodes_int=env,
                            type_int=type_int,
                            var_change_mech=var_change_mech,
                            var_change=var_change,
                        )
                        for env in envs
                    ]
                )

                x_samples_for_se = decoder @ z_samples_for_se

                x_samples_cov = utils.cov(x_samples[0, : n + d, :, 0])
                xsc_eigval, xsc_eigvec = np.linalg.eigh(x_samples_cov)
                basis_of_x_supp = xsc_eigvec[:, -n:]
                x_samples_for_se_on_x_supp = basis_of_x_supp.T @ x_samples_for_se

                hat_sx_fns = list[Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]]()
                for i in range(len(envs)):
                    # If we know the latent model is Linear Gaussian, score estimation
                    # is essentially just precision matrix --- a parameter --- estimation
                    if enable_gaussian_score_est and scm_type == "linear":
                        hat_sx_fn_i_on_x_supp = utils.gaussian_score_est(x_samples_for_se_on_x_supp[i])
                        def hat_sx_fn_i(
                            x_in: npt.NDArray[np.float_],
                            # python sucks... capture value with this since loops are NOT scopes
                            hat_sx_fn_i_on_x_supp: Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]] = hat_sx_fn_i_on_x_supp,
                        ) -> npt.NDArray[np.float_]:
                            """Reduce input down to support of x, compute estimate, transform the result back up."""
                            return basis_of_x_supp @ hat_sx_fn_i_on_x_supp(basis_of_x_supp.T @ x_in)
                    else:
                        # If not Linear Gaussian, parametric approach doesn't work, TODO: add SSM.
                        raise ValueError(f"Score estimation for {scm_type=} is not implemented!")

                    hat_sx_fns.append(hat_sx_fn_i)

                sx_samples = np.stack(
                    [
                        hat_sx_fns[env_idx](x_samples[0, ...])
                        for env_idx in range(len(envs))
                    ]
                )
                sz_samples = decoder.T @ sx_samples

            else:
                # Use ground truth score functions
                sz_samples = np.stack(
                    [
                        scm.score_fn(
                            z_samples[0, ...],
                            nodes_int=env,
                            type_int=type_int,
                            var_change_mech=var_change_mech,
                            var_change=var_change,
                        )
                        for env in envs
                    ]
                )
                sx_samples = encoder.T @ sz_samples

            dsz_samples = sz_samples[0, ...] - sz_samples[1:, ...]
            dsz_cor = (
                np.swapaxes(dsz_samples[..., 0], -1, -2) @ dsz_samples[..., 0]
            ) / nsamples

            dsx_samples = sx_samples[0, ...] - sx_samples[1:, ...]
            dsx_cor = (
                np.swapaxes(dsx_samples[..., 0], -1, -2) @ dsx_samples[..., 0]
            ) / nsamples

            ######## TEST

            # Record input state (except data samples)
            #results_run["scm"] = scm
            results_run["decoder"] = decoder
            results_run["encoder"] = encoder

            results_run["dwc"] = np.zeros((n, n))
            results_run["dws"] = np.zeros((n, n))
            results_run["hat_g_s"] = np.zeros((n, n), dtype=bool)
            results_run["hat_g_h"] = np.zeros((n, n), dtype=bool)
            results_run["dag_gt_s"] = np.zeros((n, n), dtype=int)
            results_run["dag_gt_h"] = np.zeros((n, n), dtype=int)
            results_run["hat_enc_s"] = np.zeros((n, d))
            results_run["hat_enc_h"] = np.zeros((n, d))
            results_run["pi"] = np.arange(n)

            try:
                w_mat_c, hat_enc_c, w_mat_s, hat_enc_s, hat_g_s, hard_results = umni_crl(x_samples, dsx_samples, hard_intervention, hard_graph_postprocess, ATOL_EIGV, ATOL_CI_TEST)

            except Exception as err:
                logger.error(f"Unexpected {err=}, masking entry out")
                results_run["is_run_ok"] = False
                continue

            # This run succeeded
            results_run["is_run_ok"] = True

            # Causal order step
            dwc = d_mat @ w_mat_c
            # up to ancestors step
            dws = d_mat @ w_mat_s

            # DAG targets: true DAG for hard int, transitive closure for soft int
            dag_gt_h = scm.adj_mat.astype(int)
            dag_gt_s = utils.dag.transitive_closure(dag_gt_h)
            assert dag_gt_s is not None
            all_top_orders = utils.dag.find_all_top_order(dag_gt_h)
            pi = all_top_orders[0]
            # need to check which order to compare.
            distance_to_pi = n * n
            for current_pi in all_top_orders:
                # check if dwc is upper tri
                dist = np.sum(np.tril(np.abs(dwc[current_pi]), -1))
                if dist == 0:
                    distance_to_pi = dist
                    pi = current_pi
                    break
                elif dist < distance_to_pi:
                    distance_to_pi = dist
                    pi = current_pi
                else:
                    continue

            pi_inv = np.arange(n)
            pi_inv[pi] = np.arange(n)

            dwc = dwc[pi]
            dws = dws[pi]
            hat_g_s = hat_g_s[pi_inv][:,pi_inv]
            hat_enc_c = hat_enc_c[pi_inv]
            hat_enc_s = hat_enc_s[pi_inv]


            results[n,d][run_idx]["dwc"] = dwc
            results[n,d][run_idx]["dws"] = dws
            results[n,d][run_idx]["hat_g_s"] = hat_g_s
            results[n,d][run_idx]["dag_gt_s"] = dag_gt_s

            logger.debug(
                "\nStep 2: D . W_c\n%s\n" +
                "Step 2: H_c . G\n%s\n"
                "Step 3: D . W_s\n%s\n" +
                "Step 3: H_s . G \n%s\n" +
                "True transitive closure\n%s\n" +
                "Soft graph estimate\n%s",
                dwc,
                np.round(hat_enc_c @ decoder, 4),
                dws,
                np.round(hat_enc_s @ decoder, 4),
                dag_gt_s,
                hat_g_s.astype(int),
            )

            edge_cm_s = utils.dag.confusion_mat_graph(dag_gt_s, hat_g_s)

            results[n, d][run_idx]["shd_s"]            = edge_cm_s[0][1] + edge_cm_s[1][0]
            results[n, d][run_idx]["edge_precision_s"] = (edge_cm_s[0][0] / (edge_cm_s[0][0] + edge_cm_s[0][1])) if (edge_cm_s[0][0] + edge_cm_s[0][1]) != 0 else 1.0
            results[n, d][run_idx]["edge_recall_s"]    = (edge_cm_s[0][0] / (edge_cm_s[0][0] + edge_cm_s[1][0])) if (edge_cm_s[0][0] + edge_cm_s[1][0]) != 0 else 1.0

            eff_transform_s = hat_enc_s @ decoder
            eff_transform_s *= (np.sign(np.diagonal(eff_transform_s)) / np.linalg.norm(eff_transform_s, ord=2, axis=1))[:, None]
            hat_z_samples_s = eff_transform_s @ z_samples
            results[n, d][run_idx]["norm_z_err_s"] = ((hat_z_samples_s - z_samples).__pow__(2).sum() ** (0.5)) / z_samples_norm

            soft_mixing_mat = np.eye(n,dtype=bool) | dag_gt_s.T
            results[n, d][run_idx]["extra_nz_in_eff_s"] = np.sum(
                (np.abs(eff_transform_s) >= ATOL_EFF_NZ) & ~soft_mixing_mat, dtype=int
            )



            results[n, d][run_idx]["distance_to_pi"] = distance_to_pi
            results[n, d][run_idx]["pi"] = pi

            if hard_intervention:
                assert hard_results is not None
                hat_enc_h, hat_g_h = hard_results
                # permutation
                hat_g_h = hat_g_h[pi_inv][:,pi_inv]
                hat_enc_h = hat_enc_h[pi]
                logger.debug("\nStep 4: H_h . G\n%s\n" +
                    "True DAG\n%s\n" +
                    "Hard graph estimate\n%s",
                    np.round(hat_enc_h @ decoder, 4)[pi],
                    dag_gt_h,
                    hat_g_h.astype(int),
                )

                results[n,d][run_idx]["dag_gt_h"] = dag_gt_h
                results[n,d][run_idx]["hat_g_h"] = hat_g_h

                edge_cm_h = utils.dag.confusion_mat_graph(dag_gt_h,hat_g_h)

                results[n, d][run_idx]["shd_h"]            = edge_cm_h[0][1] + edge_cm_h[1][0]
                results[n, d][run_idx]["edge_precision_h"] = (edge_cm_h[0][0] / (edge_cm_h[0][0] + edge_cm_h[0][1])) if (edge_cm_h[0][0] + edge_cm_h[0][1]) != 0 else 1.0
                results[n, d][run_idx]["edge_recall_h"]    = (edge_cm_h[0][0] / (edge_cm_h[0][0] + edge_cm_h[1][0])) if (edge_cm_h[0][0] + edge_cm_h[1][0]) != 0 else 1.0

                # The maximal theoretically allowed mixing pattern is the identity matrix.
                eff_transform_h = hat_enc_h @ decoder
                eff_transform_h *= (np.sign(np.diagonal(eff_transform_h)) / np.linalg.norm(eff_transform_h, ord=2, axis=1))[:, None]
                hat_z_samples_h = eff_transform_h @ z_samples
                results[n, d][run_idx]["norm_z_err_h"] = ((hat_z_samples_h - z_samples).__pow__(2).sum() ** (0.5)) / z_samples_norm
                results[n, d][run_idx]["extra_nz_in_eff_h"] = np.sum((np.abs(eff_transform_h) >= ATOL_EFF_NZ) & ~np.eye(n, dtype=bool), dtype=int)

    t1 = time.time() - t0
    logger.info(f"Algo finished in {t1} sec")

    # Transpose the results dict to make it more functional
    res = {
        nd: {
            k: [results_run[k] for results_run in results_run_list]
            for k in results_run_list[0].keys()
        }
        for (nd, results_run_list) in results.items()
    }

    logger.info("\n" +
        f"Results ({nruns=}, {nsamples_for_se=})\n" +
        f"    (n, d) pairs = {nd_list}, D choice = {d_mat_choice}\n" +
        f" hard int: {hard_intervention}\n" +
        f" alpha: {ATOL_CI_TEST}, eig th: {ATOL_EIGV} \n" +
        f" noisy scores: {estimate_score_fns}")

    is_run_ok = np.array([res[n, d]["is_run_ok"] for (n, d) in nd_list])
    n_ok_runs = is_run_ok.sum(-1)

    shd_s = np.array([res[n, d]["shd_s"] for (n, d) in nd_list])
    edge_precision_s = np.array([res[n, d]["edge_precision_s"] for (n, d) in nd_list])
    edge_recall_s = np.array([res[n, d]["edge_recall_s"] for (n, d) in nd_list])
    norm_z_err_s = np.array([res[n, d]["norm_z_err_s"] for (n, d) in nd_list])
    extra_nz_in_eff_s = np.array([res[n, d]["extra_nz_in_eff_s"] for (n, d) in nd_list])
    dist_to_causal_order = np.array([res[n, d]["distance_to_pi"] for (n, d) in nd_list])
    pi_all = np.array([res[n, d]["pi"] for (n, d) in nd_list])
    dwc_all = np.array([res[n, d]["dwc"] for (n, d) in nd_list])
    dws_all = np.array([res[n, d]["dws"] for (n, d) in nd_list])
    hat_g_s_all = np.array([res[n, d]["hat_g_s"] for (n, d) in nd_list])
    hat_g_h_all = np.array([res[n, d]["hat_g_h"] for (n, d) in nd_list])
    dag_gt_s_all = np.array([res[n, d]["dag_gt_s"] for (n, d) in nd_list])
    dag_gt_h_all = np.array([res[n, d]["dag_gt_h"] for (n, d) in nd_list])

    nd_number = len(nd_list)
    soft_mixing_mat_all = [[np.eye(nd_list[idx][0], dtype=bool) | g.T for g in dag_gt_s_all[idx]] for idx in range(nd_number)]
    soft_mixing_mat_all = np.asarray(soft_mixing_mat_all)

    n_zeros_in_soft_mixing_mat = [np.sum(soft_mixing_mat_all[idx] == 0) / n_ok_runs[idx]  for idx in range(nd_number)]


    incorrect_mixing_soft = extra_nz_in_eff_s.sum(-1) / n_ok_runs

    logger.info(
        f"    Ratio of failed runs = {1.0 - n_ok_runs / nruns}\n" +
        f"Up to ancestors\n" +
        f"    Structural Hamming dist = {np.around(shd_s.sum(-1) / n_ok_runs, 3)}\n" +
        f"    Edge precision          = {np.around(edge_precision_s.sum(-1) / n_ok_runs, 3)}\n" +
        f"    Edge recall             = {np.around(edge_recall_s.sum(-1) / n_ok_runs, 3)}\n" +
        f"    Normalized Z error      = {np.around(norm_z_err_s.sum(-1) / n_ok_runs, 3)}\n" +
        f"    # of incorrect mixing   = {np.around(extra_nz_in_eff_s.sum(-1) / n_ok_runs, 3)}\n" +
        f"    Ratio incorrect mixing  = {np.around(incorrect_mixing_soft /n_zeros_in_soft_mixing_mat,3)}\n" +
        f"    Dist to causal order    = {np.around(dist_to_causal_order.sum(-1) / n_ok_runs, 3)}")

    if hard_intervention:
        shd_h = np.array([res[n, d]["shd_h"] for (n, d) in nd_list])
        edge_precision_h = np.array([res[n, d]["edge_precision_h"] for (n, d) in nd_list])
        edge_recall_h = np.array([res[n, d]["edge_recall_h"] for (n, d) in nd_list])
        norm_z_err_h = np.array([res[n, d]["norm_z_err_h"] for (n, d) in nd_list])
        extra_nz_in_eff_h = np.array([res[n, d]["extra_nz_in_eff_h"] for (n, d) in nd_list])

        n_zeros_in_hard_mixing_mat = [n**2 - n for (n, d) in nd_list]
        incorrect_mixing_hard = extra_nz_in_eff_h.sum(-1) / n_ok_runs

        logger.info(
            f"Unmixing\n" +
            f"    Structural Hamming dist = {np.around(shd_h.sum(-1) / n_ok_runs, 3)}\n" +
            f"    Edge precision          = {np.around(edge_precision_h.sum(-1) / n_ok_runs, 3)}\n" +
            f"    Edge recall             = {np.around(edge_recall_h.sum(-1) / n_ok_runs, 3)}\n" +
            f"    Normalized Z error      = {np.around(norm_z_err_h.sum(-1) / n_ok_runs, 3)}\n" +
            f"    # of incorrect mixing   = {np.around(extra_nz_in_eff_h.sum(-1) / n_ok_runs, 3)}\n" +
            f"    Ratio incorrect mixing  = {np.around(incorrect_mixing_hard /n_zeros_in_hard_mixing_mat,3)}")
