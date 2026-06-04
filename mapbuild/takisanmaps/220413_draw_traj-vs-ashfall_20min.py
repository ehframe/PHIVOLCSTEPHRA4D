# 任意の噴火のリリースポイント
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gs
import glob
from scipy import stats
import itertools
from scipy import integrate
from template import tgsd_costa2016 as tgsd_c16

# direc1 = "C:/Users/theta/OneDrive - Kyoto University/labostorage/mydata/90-pyfile/2022_multi-emission_20min/"
direc2 = 'F:/Tephra4D/'
demfilename = '../../14-DEM/SakuraDEM.csv'
dem = pd.read_csv(demfilename, header=0, index_col=0)
pallet = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]
# fp = FontProperties(fname='C:/Windows/Fonts/UDDigiKyokashoN-R.ttc', size=11)
# fp = FontProperties(fname='C:/Windows/Fonts/UDDigiKyokashoN-R.ttc', size=9.5)
fp = FontProperties(fname='C:/Windows/Fonts/FOT-SkipProN-D.otf', size=11)
fp2 = FontProperties(fname='C:/Windows/Fonts/FOT-SkipProN-D.otf', size=9.5)
site_calc = pd.read_csv('../inpfile/forwardpoint-cross300m_forcalc.csv', header=0)
site_calc["h"] = site_calc["h"] // 10 * 10

table_det = pd.read_csv("../inpfile/detect_list.csv", index_col=0, header=0).fillna(0)
table_er = pd.read_csv("../../13-JMA/eruptlist/JMAexer_190601-210731.csv", index_col=0, header=0).fillna(0)
x_int = 10.546
y_int = 12.325
x = np.arange(-7354, 1342 * x_int - 7354, x_int)  # x軸の描画範囲の生成。0から1125まで1刻み。
y = np.arange(862 * y_int - 4521, -4521, -y_int)  # y軸の描画範囲の生成。0から750まで1刻み。
X, Y = np.meshgrid(x, y)
parsivel_site = pd.DataFrame(index=['h', 'd', 'x', 'y'],
                             data={"SVO": [30, 5454, -5359, 1018],
                                   "HAR": [410, 2708, -2295, 1448],
                                   "FUT": [5, 5023, 59, 5022],
                                   "KOM": [135, 4658, 2660, 3824],
                                   "KUR": [65, 4131, 4117, 339],
                                   "ARI": [105, 2562, 494, -2514],
                                   "SBT": [120, 3044, -2146, -2159],
                                   "AKA": [10, 4541, -4494, -655],
                                   "NAB": [200, 2844, 2486, -1344],
                                   "NTT": [10, 5121, -4274, 2820],
                                   "SAI": [25, 4852, -2207, 4321],
                                   "URA": [5, 5666, 4804, 3004],
                                   "HIK": [545, 1582, -1532, 396],
                                   "MAT": [320, 2950, -245, 2940],
                                   "HKU": [235, 3035, -2971, 623],
                                   "HKD": [65, 4387, -4290, 915],
                                   "ART": [80, 2726, 1516, -2266],
                                   "JIGK": [125, 2995, 2888, 794],
                                   "MOCK": [305, 1995, -1635, -1144],
                                   "FUR2": [200, 2378, -851, -2221],
                                   "KIT": [645, 2051, 557, 1973]}).T
vent_z = 1000
h_slice = 100
time_interval = 20  # min
g = 9.81
mu = 1.8e-5
rho_a = 1.293
rho_p = 2640
vel = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5, 1.7, 1.9,
                2.2, 2.6, 3, 3.4, 3.8, 4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12, 13.6, 15.2, 17.6, 20.8])
diameter = pd.Series(
    [6.41, 4.76, 4.24, 3.92, 3.70, 3.52, 3.37, 3.22, 3.10, 2.98, 2.87, 2.67, 2.48, 2.31, 2.15, 1.99, 1.70, 1.44, 1.19,
     0.97, 0.76, 0.39, 0.06, -0.24, -0.51, -0.76, -1.23, -1.65, -2.03, -2.66, -3.19, -3.63],
    index=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8,
           5.6, 6.4, 7.2, 8, 9.6, 11.2, 12.8, 16, 19.2, 22.4])


def draw_traj_vs_ashfall(erno):
    p_valid = table_det.columns[1:][
        [table_det.loc[er, table_det.columns[i]] not in ["x", "m"] for i in range(1, len(table_det.columns))]]
    ertime = pd.to_datetime(table_det.loc[erno, "ertime"])
    mh = table_er.loc[erno, "h_p"]
    minutes = 240
    obs_sphe = pd.read_csv("obs_er" + str(erno) + "_time_mfilt_theta.csv")  # (direc2 + "w_rate/obs_er" + str(erno) + "_time_mfilt_theta.csv")
    obsall = pd.DataFrame(index=p_valid, columns=vel[::4]).fillna(0)
    p_detected = np.unique(obs_sphe["site"])
    for p in p_detected:
        obsall.loc[p, :] = np.array([np.sum(
            obs_sphe[(obs_sphe["site"] == p) & (obs_sphe["min"] <= minutes)].iloc[:, 4 + sl * 4:8 + sl * 4].values) for
            sl in range(8)])
    sheet2 = pd.DataFrame(columns=np.concatenate([["point", "seg"], vel[::4]]))

    def f_obsvalue(iname):
        if iname in p_detected:
            unchi2 = obsall[obsall.index == iname]
            if len(unchi2) == 0:
                obsvalue = np.zeros(8)
            else:
                obsvalue = unchi2.iloc[0, :].values
        else:
            obsvalue = np.zeros(8)
        obsvalue = obsvalue.tolist()
        return obsvalue

    obs_table = np.array(list(map(f_obsvalue, p_valid)))
    obs_table = pd.DataFrame(np.where(obs_table == 0, 1e-6, obs_table), index=p_valid, columns=sheet2.columns[2:])
    # obs_norm = obs_table / np.max(obs_table)
    # obs_threshold = np.min(obs_norm)

    plt.close()
    fig = plt.figure(figsize=(10, 6), dpi=200)
    g_fig = gs.GridSpec(9, 6)
    parsivel = parsivel_site.loc[p_valid, :]
    for i_20min in range(6):
        trajlist = glob.glob(str(erno) + "/" + '*mms-1_er' + str(erno) + '_' + str(i_20min * 20) + 'min_220406.csv')  # direc2 + "trajfile/" + '*mms-1_er' + str(erno) + '_' + str(i_20min * 20) + '_220406.csv')
        dirlist = (np.array(
            [trajlist[i].replace(str(erno) + "\\", "").replace('mms-1_er' + str(erno) + '_' +
                                                                    str(i_20min * 20) + 'min_220406.csv', "") for i in
             range(len(trajlist))]).astype(float)) / 1000
        for i_vt in range(1, 5):
            ax = fig.add_subplot(g_fig[(i_vt - 1) * 2: (i_vt - 1) * 2 + 2, i_20min:i_20min + 1])
            for p in parsivel[obs_table[str(vel[i_vt * 4])] > 1].index:
                plt.plot(parsivel.loc[p, "x"], parsivel.loc[p, "y"], linewidth=0, marker="o", markersize=5,
                         markeredgecolor="k", markerfacecolor=pallet[6], zorder=5)
            for order in range(6):
                for p in parsivel[(obs_table[str(vel[i_vt * 4])] < 10 ** -order) &
                                  (obs_table[str(vel[i_vt * 4])] > 10 ** (-order - 1))].index:
                    plt.plot(parsivel.loc[p, "x"], parsivel.loc[p, "y"], linewidth=0, marker="o", markersize=5,
                             markeredgecolor="k", markerfacecolor=pallet[5 - order], zorder=5)
            for p in parsivel[obs_table[str(vel[i_vt * 4])] <= 1e-6].index:
                plt.plot(parsivel.loc[p, "x"], parsivel.loc[p, "y"], linewidth=0, marker="o", markersize=5,
                         markeredgecolor="k", markerfacecolor="k", zorder=5)
            for i_i_vt in range(4):
                traj = pd.read_csv(str(erno) + "/" + str(  # (direc2 + "trajfile/" + str(
                    int(np.min(dirlist[dirlist >= vel[i_vt * 4 + i_i_vt]]) * 1000)) + "mms-1_er" + str(erno) +
                                   '_' + str(i_20min * 20) + 'min_220406.csv', index_col=None)
                for h_seg in np.arange(1100, mh + 1100, 100):
                    traj_hfilt = traj[(traj["d"] == vel[i_vt * 4 + i_i_vt]) & (traj["h"] == h_seg)]
                    plt.plot(traj_hfilt.iloc[:, 2] - 657324, traj_hfilt.iloc[:, 3] - 3495137, linewidth=0.5,
                             marker=None, color=pallet[np.min([(h_seg - 1000) // 500, 5])],
                             linestyle=["solid", "dashed", "dotted", "dashdot"][i_i_vt])
            ax.contourf(X, Y, dem, levels=[0, 2000], colors=['0.9'], zorder=0)
            # plt.contour(X, Y, dem, levels=range(200, 1100, 200), colors=['0.6'], linewidths=0.5, zorder=1)
            plt.contour(X, Y, dem, levels=[0], colors=['k'], linewidths=0.5, zorder=1)
            ax.set_aspect("equal")
            if i_20min == 0:
                plt.ylabel(str(vel[i_vt * 4]) + "-" + str(vel[i_vt * 4 + 3]) + "m/s", fontproperties=fp2)
            if i_vt == 1:
                plt.title(str(i_20min * 20) + "min", fontproperties=fp2)
            plt.xlim([-7354, 14142 - 7354])
            plt.ylim([-4521, 10611 - 4521])
            plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
        ax2 = fig.add_subplot(g_fig[-1, 2:4])
        ax2.set_ylim(0, 1)
        for i in range(8):
            if i == 0:
                ax2.fill([i, i + 1, i + 1, i], [0, 0, 1, 1], color="0.6")
            else:
                ax2.fill([i, i + 1, i + 1, i], [0, 0, 1, 1], color=pallet[i - 1])
        plt.xlim([0, 6])
        plt.xticks(range(9),
                   [0, "10$^{−6}$", "10$^{−5}$", "10$^{−4}$", "10$^{−3}$", "10$^{−2}$", "10$^{−1}$", "10$^0$", "10"],
                   fontproperties=fp2)
        plt.xlabel("L (kg/m$^2$)", fontproperties=fp)
        plt.tick_params(labelleft=False, left=False)
    plt.suptitle(ertime.strftime("%Y/%m/%d %H:%M h$_p$=") + str(mh) + "m", fontproperties=fp)
    plt.subplots_adjust(hspace=0.2)
    plt.savefig(str(erno) + "_traj_20min.png", bbox_inches='tight')  # direc2 + "trajfig/" + str(erno) + "_traj_20min.png", bbox_inches='tight')


for er in [19147]:  # table_det.index[:55]:
    draw_traj_vs_ashfall(er)

