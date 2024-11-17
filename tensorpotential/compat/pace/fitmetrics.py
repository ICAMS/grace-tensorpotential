import numpy as np


class FitMetrics:
    def __init__(self, w_e, w_f, e_scale, f_scale, ncoefs, regs=None):
        self.w_e = w_e
        self.w_f = w_f
        self.e_scale = e_scale
        self.f_scale = f_scale
        self.regs = regs
        self.ncoefs = ncoefs
        self.nfuncs = None
        self.time_history = []

        self.loss = 0
        self.eval_time = 0

    def record_time(self, time):
        self.time_history.append(time)

    def to_FitMetricsDict(self):
        """
        Store all metric-relevant info into a dictionary
        :return: fit metrics dictionary
        """

        regularization_loss = [
            float(r_comp * r_weight)
            for r_comp, r_weight in zip(self.regs, self.reg_weights)
        ]
        l1 = regularization_loss[0]
        l2 = regularization_loss[1]
        smoothness_reg_loss = regularization_loss[2:]
        res_dict = {
            # total loss
            "loss": self.loss,
            # loss contributions
            "e_loss_contrib": self.e_loss * self.e_scale,
            "f_loss_contrib": self.f_loss * self.f_scale,
            "l1_reg_contrib": l1,
            "l2_reg_contrib": l2,
            "extra_regularization_contrib": smoothness_reg_loss,
            # non-weighted e and f losses
            "e_loss": self.e_loss,
            "f_loss": self.f_loss,
            # e and f loss weights (scales)
            "e_scale": self.e_scale,
            "f_scale": self.f_scale,
            # RMSE metrics
            "rmse_epa": self.rmse_epa,
            "low_rmse_epa": self.low_rmse_epa,
            "rmse_f": self.rmse_f,
            "low_rmse_f": self.low_rmse_f,
            "rmse_f_comp": self.rmse_f_comp,
            "low_rmse_f_comp": self.low_rmse_f_comp,
            # MAE metrics
            "mae_epa": self.mae_epa,
            "low_mae_epa": self.low_mae_epa,
            "mae_f": self.mae_f,
            "low_mae_f": self.low_mae_f,
            "mae_f_comp": self.mae_f_comp,
            "low_mae_f_comp": self.low_mae_f_comp,
            # MAX metrics
            "max_abs_epa": self.max_abs_epa,
            "low_max_abs_epa": self.low_max_abs_epa,
            "max_abs_f": self.max_abs_f,
            "low_max_abs_f": self.low_max_abs_f,
            "max_abs_f_comp": self.max_abs_f_comp,
            "low_max_abs_f_comp": self.low_max_abs_f_comp,
            "eval_time": self.eval_time,
            "nat": self.nat,
            "ncoefs": self.ncoefs,
        }

        if self.nfuncs is not None:
            res_dict["nfuncs"] = self.nfuncs
        return res_dict

    def from_FitMetricsDict(self, fit_metrics_dict):
        self.loss = fit_metrics_dict["loss"]

        self.e_loss = fit_metrics_dict["e_loss"]
        self.f_loss = fit_metrics_dict["f_loss"]

        self.e_scale = fit_metrics_dict["e_scale"]
        self.f_scale = fit_metrics_dict["f_scale"]

        # RMSE metrics
        self.rmse_epa = fit_metrics_dict["rmse_epa"]
        self.low_rmse_epa = fit_metrics_dict["low_rmse_epa"]
        self.rmse_f = fit_metrics_dict["rmse_f"]
        self.low_rmse_f = fit_metrics_dict["low_rmse_f"]

        self.rmse_f_comp = fit_metrics_dict["rmse_f_comp"]
        self.low_rmse_f_comp = fit_metrics_dict["low_rmse_f_comp"]

        # MAE metrics
        self.mae_epa = fit_metrics_dict["mae_epa"]
        self.low_mae_epa = fit_metrics_dict["low_mae_epa"]
        self.mae_f = fit_metrics_dict["mae_f"]
        self.low_mae_f = fit_metrics_dict["low_mae_f"]
        self.mae_f_comp = fit_metrics_dict["mae_f_comp"]
        self.low_mae_f_comp = fit_metrics_dict["low_mae_f_comp"]

        # MAX metrics
        self.max_abs_epa = fit_metrics_dict["max_abs_epa"]
        self.low_max_abs_epa = fit_metrics_dict["low_max_abs_epa"]
        self.max_abs_f = fit_metrics_dict["max_abs_f"]
        self.low_max_abs_f = fit_metrics_dict["low_max_abs_f"]

        self.max_abs_f_comp = fit_metrics_dict["max_abs_f_comp"]
        self.low_max_abs_f_comp = fit_metrics_dict["low_max_abs_f_comp"]

        self.eval_time = fit_metrics_dict["eval_time"]
        self.nat = fit_metrics_dict["nat"]
        self.ncoefs = fit_metrics_dict["ncoefs"]

        if "nfuncs" in fit_metrics_dict:
            self.nfuncs = fit_metrics_dict["nfuncs"]

    def compute_metrics(self, de, de_pa, df, nat, dataframe=None, de_low=None):
        if de_low is None:
            de_low = 1.0
        self.nat = np.sum(nat)
        self.rmse_epa = np.sqrt(np.mean(de_pa**2))
        self.rmse_e = np.sqrt(np.mean(de**2))
        self.rmse_f = np.sqrt(np.mean(np.sum(df**2, axis=1)))
        self.rmse_f_comp = np.sqrt(np.mean(df**2))  # per component
        self.mae_epa = np.mean(np.abs(de_pa))
        self.mae_e = np.mean(np.abs(de))
        self.mae_f = np.mean(np.linalg.norm(df, axis=1))
        self.mae_f_comp = np.mean(np.abs(df).flatten())  # per component
        self.mae_f = np.mean(np.sum(np.abs(df), axis=1))
        # self.mae_f = np.mean(np.linalg.norm(df, axis=1))

        self.e_loss = float(np.sum(self.w_e * de_pa**2))
        self.f_loss = np.sum(self.w_f * df**2)
        self.max_abs_e = np.max(np.abs(de))
        self.max_abs_epa = np.max(np.abs(de_pa))
        self.max_abs_f = np.max(np.abs(df))
        self.max_abs_f_comp = np.max(np.abs(df).flatten())  # per component

        self.low_rmse_epa = 0
        self.low_mae_epa = 0
        self.low_max_abs_epa = 0
        self.low_rmse_f = 0
        self.low_mae_f = 0
        self.low_max_abs_f = 0
        self.low_rmse_f_comp = 0
        self.low_mae_f_comp = 0

        if dataframe is not None:
            try:
                if "e_chull_dist_per_atom" in dataframe.columns:
                    nrgs = (
                        dataframe["e_chull_dist_per_atom"]
                        .to_numpy()
                        .reshape(
                            -1,
                        )
                    )
                    mask = nrgs <= de_low
                else:
                    nrgs = dataframe["energy_corrected"].to_numpy().reshape(
                        -1,
                    ) / nat.reshape(
                        -1,
                    )
                    emin = min(nrgs)
                    mask = nrgs <= (emin + de_low)
                mask_f = np.repeat(
                    mask,
                    nat.reshape(
                        -1,
                    ),
                )
                self.low_rmse_epa = np.sqrt(np.mean(de_pa[mask] ** 2))
                self.low_mae_epa = np.mean(np.abs(de_pa[mask]))
                self.low_max_abs_epa = np.max(np.abs(de_pa[mask]))
                self.low_rmse_f = np.sqrt(np.mean(np.sum(df[mask_f] ** 2, axis=1)))
                self.low_mae_f = np.mean(np.linalg.norm(df[mask_f], axis=1))
                self.low_max_abs_f = np.max(np.abs(df[mask_f]))
                self.low_rmse_f_comp = np.sqrt(
                    np.mean(df[mask_f] ** 2)
                )  # per component
                self.low_mae_f_comp = np.mean(
                    np.abs(df[mask_f]).flatten()
                )  # per component
                self.low_max_abs_f_comp = np.max(
                    np.abs(df[mask_f]).flatten()
                )  # per component
            except:
                pass
