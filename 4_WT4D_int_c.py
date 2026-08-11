import numpy as np
import pandas as pd
import tgsd_costa2016 as tgsd_c16
from scipy import integrate



vent_x = 657324  # 295464
vent_y = 3495137  # 3913472
vent_z = 1000  # 2590
time_interval = 180  # min
h_slice = 500
ertime = pd.to_datetime("2026/7/09 00:00")
erno = 2026709
vt_ser = pd.Series([12.535, 8.694, 5.693, 3.5, 1.962, 0.864, 0.267, 0.069, 0.017, 0.004],
                   index=np.arange(-2, 8))
time_range = 480  # range of calculation time
dur = 2  # duration of constant plume


dir = '/home/ehf/PHIVOLCSTEPHRA4D/2026709_files/'
dir1 = dir + str(erno)  + "/w_rate/"
LOAD_EPS = 1e-20


def temp_tsp_slice(mh):
    # Suzuki 1983 distribution
    # traj = pd.read_csv("1_traj_rise_" + ertime.strftime('%Y%m%d%H%M') + "_WT4D10km_20min.csv", index_col=None)
    # mh = np.max(traj["z"]) - vent_z
    # 時間高度プロファイルを作る (time height profile)
    h_ar = np.arange(h_slice, mh + h_slice, h_slice)
    # 噴出量と高度の関係を求める
    mer = 5e5 / 20  # t/min
    # 噴出量と高度の関係を求める
    k = 8
    q_z = np.zeros(len(h_ar))

    def den_func(z):
        if z <= mh:
            den_z = k ** 2 * (1 - z / mh) * np.exp(-k * (1 - z / mh)) / (mh * (1 - (1 + k) * np.exp(-k)))
        else:
            den_z = 0
        return den_z

    tot = integrate.quad(den_func, 0, h_ar[-1] + h_slice)[0]
    q_z += [np.max([np.round(mer * integrate.quad(den_func, z - h_slice, z)[0] / tot, 3), 0]) for z in h_ar]
    den = pd.Series(q_z, index=h_ar)
    return den


def f_int_c_tz(ertime):
    w_ratefilename = dir1 + "w_rate_er" + str(erno) + "_C004_site.csv"
    tpointfilename = dir1 +  "tpoint_er" + str(erno) + "_C004_site.csv"
    w_rate = pd.read_csv(w_ratefilename)
    tpoint = pd.read_csv(tpointfilename)
    w_rate = w_rate.set_index(["v_t", "site"])
    tpoint = tpoint.set_index(["v_t", "site"])
    w_rate.columns = w_rate.columns.astype(float)
    tpoint.columns = tpoint.columns.astype(float)
    if len(w_rate) == 0:
        return pd.DataFrame()
    mh = int(np.max(w_rate.columns) - vent_z)
    if mh <= 0:
        return pd.DataFrame()
    den = temp_tsp_slice(mh)
    den.index = den.index.astype(int)
    den_abs = pd.Series(
        np.interp(w_rate.columns.to_numpy(), den.index.to_numpy() + vent_z, den.to_numpy(), left=0, right=0),
        index=w_rate.columns,
    )
    eta = 8
    tgsd = tgsd_c16.tgsd_func(eta, np.max(den.index) / 1000)
    tgsd = tgsd.abs()
    if float(tgsd.sum()) > 0:
        tgsd = tgsd / float(tgsd.sum())
    tgsd.columns = [0]
    conc_tz = pd.DataFrame(np.zeros((len(np.unique(w_rate.index.get_level_values("site"))), time_range)),
                           index=np.unique(w_rate.index.get_level_values("site")), columns=np.arange(time_range))
    df2 = pd.DataFrame()
    vt_vals = w_rate.index.get_level_values("v_t").to_numpy()
    tgsd_vt = tgsd.reindex(vt_vals, method="nearest").to_numpy()
    load_site = pd.DataFrame(np.array([tgsd_vt]).T * w_rate *
                             den_abs.values, index=w_rate.index)  # kg
    print("DEBUG f_int_c_tz: w_rate min/max =", float(w_rate.to_numpy().min()), float(w_rate.to_numpy().max()))
    print("DEBUG f_int_c_tz: den_abs min/max =", float(den_abs.min()), float(den_abs.max()))
    print("DEBUG f_int_c_tz: tgsd_vt min/max =", float(np.min(tgsd_vt)), float(np.max(tgsd_vt)))
    print("DEBUG f_int_c_tz: load_site min/max =", float(load_site.to_numpy().min()), float(load_site.to_numpy().max()))
    tpoint_site = pd.DataFrame(np.where(load_site < LOAD_EPS, 0, tpoint / 60),
                               index=tpoint.index, columns=tpoint.columns)
    load_site = pd.DataFrame(
        np.where(load_site < LOAD_EPS, 0, load_site),
        index=load_site.index,
        columns=load_site.columns,
    )
    tpoint_site = tpoint_site.iloc[np.where(np.sum(load_site, axis=1) > 0)[0]]
    load_site = load_site.iloc[np.where(np.sum(load_site, axis=1) > 0)[0]]
    if len(load_site) > 0:
        ind = np.where(load_site > 0)
        times = tpoint_site.to_numpy()[ind]
        loads = load_site.to_numpy()[ind]
        df2 = pd.concat([df2, pd.DataFrame({
            "v_t": vt_vals[ind[0]],
            "h_seg": load_site.columns.to_numpy()[ind[1]],
            "time": times,
            "site": load_site.index.get_level_values("site").to_numpy()[ind[0]],
            "load": loads,
        })])
    if len(df2) > 0:
        df2 = df2[df2["load"] > 0]
        for i_df2 in df2.index:
            if df2.loc[i_df2, "time"] > time_range:
                continue
            for n_t in range(int(df2.loc[i_df2, "time"] - h_slice / 2 / df2.loc[i_df2, "v_t"] / 60),
                             int(np.ceil(df2.loc[i_df2, "time"] + h_slice / 2 / df2.loc[i_df2, "v_t"] / 60 + dur))):
                if n_t not in conc_tz.columns:
                    continue
                elif n_t <= df2.loc[i_df2, "time"] - h_slice / 2 / df2.loc[i_df2, "v_t"] / 60:
                    conc_tz.loc[df2.loc[i_df2, "site"], n_t] += df2.loc[i_df2, "load"] * 20 * (
                            n_t + 1 - (df2.loc[i_df2, "time"] - h_slice / 2 / df2.loc[i_df2, "v_t"] / 60)) / (
                                                                        dur + h_slice / df2.loc[i_df2, "v_t"] / 60)
                elif n_t + 1 >= df2.loc[i_df2, "time"] + h_slice / 2 / df2.loc[i_df2, "v_t"] / 60 + dur:
                    conc_tz.loc[df2.loc[i_df2, "site"], n_t] += df2.loc[i_df2, "load"] * 20 * (df2.loc[i_df2, "time"] +
                                                                                               h_slice / 2 / df2.loc[
                                                                                                   i_df2, "v_t"] / 60 + dur - n_t) / (
                                                                            dur + h_slice / df2.loc[i_df2, "v_t"] / 60)
                else:
                    conc_tz.loc[df2.loc[i_df2, "site"], n_t] += df2.loc[i_df2, "load"] * 20 / (
                            dur + h_slice / df2.loc[i_df2, "v_t"] / 60)
        return conc_tz
    return conc_tz


def f_int_c_tz_vt(ertime):
    w_ratefilename = dir1 + "w_rate_er" + str(erno) + "_C004_site.csv"
    tpointfilename = dir1 + "tpoint_er" + str(erno) + "_C004_site.csv"
    w_rate = pd.read_csv(w_ratefilename)
    tpoint = pd.read_csv(tpointfilename)
    w_rate = w_rate.set_index(["v_t", "site"])
    tpoint = tpoint.set_index(["v_t", "site"])
    w_rate.columns = w_rate.columns.astype(float)
    tpoint.columns = tpoint.columns.astype(float)
    if len(w_rate) == 0:
        return pd.DataFrame()
    mh = int(np.max(w_rate.columns) - vent_z)
    if mh <= 0:
        return pd.DataFrame()
    den = temp_tsp_slice(mh)
    den.index = den.index.astype(int)
    den_abs = pd.Series(
        np.interp(w_rate.columns.to_numpy(), den.index.to_numpy() + vent_z, den.to_numpy(), left=0, right=0),
        index=w_rate.columns,
    )
    eta = 8
    tgsd = tgsd_c16.tgsd_func(eta, np.max(den.index) / 1000)
    tgsd = tgsd.abs()
    if float(tgsd.sum()) > 0:
        tgsd = tgsd / float(tgsd.sum())
    tgsd.columns = [0]
    load_ts = pd.DataFrame()
    df2 = pd.DataFrame()
    vt_vals = w_rate.index.get_level_values("v_t").to_numpy()
    tgsd_vt = tgsd.reindex(vt_vals, method="nearest").to_numpy()
    load_site = pd.DataFrame(
        np.array([tgsd_vt]).T * w_rate * den_abs.values, index=w_rate.index)  # kg
    print("DEBUG f_int_c_tz_vt: w_rate min/max =", float(w_rate.to_numpy().min()), float(w_rate.to_numpy().max()))
    print("DEBUG f_int_c_tz_vt: den_abs min/max =", float(den_abs.min()), float(den_abs.max()))
    print("DEBUG f_int_c_tz_vt: tgsd_vt min/max =", float(np.min(tgsd_vt)), float(np.max(tgsd_vt)))
    print("DEBUG f_int_c_tz_vt: load_site min/max =", float(load_site.to_numpy().min()), float(load_site.to_numpy().max()))
    # 連続噴火は120分後まで考慮しており，最大で73分間隔だったことにより廃止
    tpoint_site = pd.DataFrame(np.where(load_site < LOAD_EPS, 0, tpoint / 60), index=tpoint.index,
                               columns=tpoint.columns)
    load_site = pd.DataFrame(
        np.where(load_site < LOAD_EPS, 0, load_site),
        index=load_site.index,
        columns=load_site.columns,
    )
    tpoint_site = tpoint_site.iloc[np.where(np.sum(load_site, axis=1) > 0)[0]]
    load_site = load_site.iloc[np.where(np.sum(load_site, axis=1) > 0)[0]]

    if len(load_site) > 0:
        ind = np.where(load_site > 0)
        times = tpoint_site.to_numpy()[ind]
        loads = load_site.to_numpy()[ind]
        df2 = pd.concat([df2, pd.DataFrame({
            "v_t": vt_vals[ind[0]],
            "h_seg": load_site.columns.to_numpy()[ind[1]],
            "time": times,
            "site": load_site.index.get_level_values("site").to_numpy()[ind[0]],
            "load": loads,
        })])
    conc_tz_vt = pd.DataFrame(np.zeros((len(np.unique(np.array([w_rate.index.get_level_values("site"),
                                                                vt_vals]).T, axis=0).T[0]),
                                        time_range)),
                              index=pd.MultiIndex.from_arrays(
                                  np.unique(np.array([w_rate.index.get_level_values("site"),
                                                      vt_vals]).T, axis=0).T,
                                  names=["site", "v_t"]), columns=np.arange(time_range))
    if len(df2) > 0:
        df2 = df2[df2["load"] > 0]
        for i_df2 in range(len(df2)):
            row = df2.iloc[i_df2]
            vt = float(row["v_t"])
            time_val = float(row["time"])
            site_val = row["site"]
            load_val = float(row["load"])
            if time_val > time_range:
                continue
            for n_t in range(int(time_val - h_slice / 2 / vt / 60),
                             int(np.ceil(time_val + h_slice / 2 / vt / 60 + dur))):
                if n_t not in conc_tz_vt.columns:
                    continue
                elif n_t <= time_val - h_slice / 2 / vt / 60:
                    conc_tz_vt.loc[(site_val, vt), n_t] += load_val * 20 * (
                            n_t + 1 - (time_val - h_slice / 2 / vt / 60)) / (
                                                                 dur + h_slice / vt / 60)
                elif n_t + 1 >= time_val + h_slice / 2 / vt / 60 + dur:
                    conc_tz_vt.loc[(site_val, vt), n_t] += load_val * 20 * (
                                time_val + h_slice / 2 / vt / 60 + dur - n_t) / (
                                                                            dur + h_slice / vt / 60)
                else:
                    conc_tz_vt.loc[(site_val, vt), n_t] += load_val * 20 / (
                            dur + h_slice / vt / 60)
        return conc_tz_vt
    return conc_tz_vt


if __name__ == '__main__':
    load_ts = f_int_c_tz(ertime)
    load_ts.to_csv(dir + "4_load_ts.csv")
    load_ts_vt = f_int_c_tz_vt(ertime)
    load_ts_vt.to_csv(dir + "4_load_ts_vt.csv")
