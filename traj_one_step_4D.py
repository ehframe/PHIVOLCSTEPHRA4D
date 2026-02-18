import numpy as np


def ttb(u1, u2, d_, c1, c2, c3, c4):  # traj_time_to_border
    if np.round(u1 * d_ - (u1 * c2 + u2 * c4) * c2, 4) == 0:
        return np.inf
    else:
        return np.ceil(
            (c1 * d_ - (c1 * c2 + c3 * c4) * c2) / (u1 * d_ - (u1 * c2 + u2 * c4) * c2) * 100) / 100


def func(u, v, w, x1, x2, x3, x4, y1, y2, y3, y4, Z, d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n):
    #   3-------4
    #   |   0   |
    #   1-------2
    x01, x13, y01, y13, x12, y12 = np.round([x1 - x0, x3 - x1, y1 - y0, y3 - y1, x2 - x1, y2 - y1], 4)
    d13, d12 = x13 ** 2 + y13 ** 2, x12 ** 2 + y12 ** 2
    if x13 == 0:
        xt1 = np.ceil(x01 / u * 100) / 100
        if xt1 > 0:
            [xt, x1n] = [xt1, x0n - 1]
        else:
            [x02, x24, y02, y24] = [x2 - x0, x4 - x2, y2 - y0, y4 - y2]
            d24 = x24 ** 2 + y24 ** 2
            if x24 == 0:
                xt3 = np.ceil(x02 / u * 100) / 100
            else:
                xt3 = ttb(u, v, d24, x02, x24, y02, y24)
                xt4 = ttb(v, u, d24, y02, y24, x02, x24)
                if xt4 != xt3:
                    print("d: " + str(d) + "h: " + str(h) + "xt3: " + str(xt3) + " xt4: " + str(xt4) + " unchi")
                    return np.zeros(19)
                # elif xt3 == 0:
                #     [xt, x1n] = [float("inf"), x0n]
                # elif xt3 < 0:
                #     if xt1 == 0:
                #         [xt, x1n] = [float("inf"), x0n]
                #     else:
                #         return np.zeros(19)
            if xt3 == 0:
                [xt, x1n] = [float("inf"), x0n]
            elif xt3 < 0:
                if xt1 == 0:
                    [xt, x1n] = [float("inf"), x0n]
                else:
                    return np.zeros(19)
            else:
                [xt, x1n] = [xt3, x0n + 1]

    else:
        xt1 = ttb(u, v, d13, x01, x13, y01, y13)
        if xt1 > 0:
            xt2 = ttb(v, u, d13, y01, y13, x01, x13)
            if xt1 != xt2:
                if np.round(np.abs(xt1 - xt2), 3) > np.max([xt1 * 1e-2, 0.01]):
                    print("d: " + str(d) + "h: " + str(h) + "xt1: " + str(xt1) + " xt2: " + str(xt2) + " unchi")
                    return np.zeros(19)
                else:
                    [xt, x1n] = [np.max([xt1, xt2]), x0n - 1]
            else:
                [xt, x1n] = [xt1, x0n - 1]
        else:
            x02, x24, y02, y24 = np.round([x2 - x0, x4 - x2, y2 - y0, y4 - y2], 4)
            d24 = x24 ** 2 + y24 ** 2
            xt3 = ttb(u, v, d24, x02, x24, y02, y24)
            xt4 = ttb(v, u, d24, y02, y24, x02, x24)
            if xt4 != xt3:
                if np.round(np.abs(xt3 - xt4), 3) > np.max([xt3 * 1e-2, 0.01]):
                    print("d: " + str(d) + "h: " + str(h) + "xt3: " + str(xt3) + " xt4: " + str(xt4) + " unchi")
                    return np.zeros(19)
                else:
                    [xt, x1n] = [np.max([xt1, xt3, xt4]),
                                 np.array([x0n - 1, x0n + 1])[[xt1, xt3] == np.max([xt1, xt3])][0]]
            else:
                [xt, x1n] = [np.max([xt1, xt3]), np.array([x0n - 1, x0n + 1])[[xt1, xt3] == np.max([xt1, xt3])][0]]
    yt = float("inf")
    y1n = y0n
    if y12 == 0:
        yt1 = np.ceil(y01 / v * 100) / 100
        if yt1 > 0:
            [yt, y1n] = [yt1, y0n - 1]
        else:
            [x03, x34, y03, y34] = [x3 - x0, x4 - x3, y3 - y0, y4 - y3]
            d34 = x34 ** 2 + y34 ** 2
            if y34 == 0:
                yt3 = np.ceil(y03 / v * 100) / 100
                if yt3 > 0:
                    [yt, y1n] = [yt3, y0n + 1]
                elif yt3 == 0:
                    [yt, y1n] = [float("inf"), y0n]
                elif yt3 < 0:
                    if yt1 == 0:
                        [yt, y1n] = [float("inf"), y0n]
                    else:
                        return np.zeros(19)
            else:
                yt3 = ttb(u, v, d34, x03, x34, y03, y34)
                yt4 = ttb(v, u, d34, y03, y34, x03, x34)
                if yt4 != yt3:
                    if np.round(np.abs(yt3 - yt4), 3) > np.max([yt3 * 1e-5, 0.01]):
                        if yt3 > xt:
                            yt = yt3
                            y1n = y0n
                        else:
                            print("d: " + str(d) + "h: " + str(h) + "yt3: " + str(yt3) + " yt4: " + str(yt4) + " unchi")
                            return np.zeros(19)
                    else:
                        [yt, y1n] = [np.max([yt3, yt4]), y0n + 1]
                elif yt3 == 0:
                    [yt, y1n] = [float("inf"), y0n]
                elif yt3 < 0:
                    if yt1 == 0:
                        [yt, y1n] = [float("inf"), y0n]
                    else:
                        return np.zeros(19)
                else:
                    [yt, y1n] = [yt4, y0n + 1]
    else:
        yt1 = ttb(u, v, d12, x01, x12, y01, y12)
        if yt1 > 0:
            yt2 = ttb(v, u, d12, y01, y12, x01, x12)
            if yt1 != yt2:
                if np.round(np.abs(yt1 - yt2), 3) > np.max([yt1 * 1e-2, 0.01]):
                    print("d: " + str(d) + "h: " + str(h) + "yt1: " + str(yt1) + " yt2: " + str(yt2) + " unchi")
                    return np.zeros(19)
                else:
                    [yt, y1n] = [np.max([yt1, yt2]), y0n - 1]
            else:
                [yt, y1n] = [yt1, y0n - 1]
        else:
            x03, x34, y03, y34 = np.round([x3 - x0, x4 - x3, y3 - y0, y4 - y3], 4)
            d34 = x34 ** 2 + y34 ** 2
            yt3 = ttb(u, v, d34, x03, x34, y03, y34)
            yt4 = ttb(v, u, d34, y03, y34, x03, x34)
            if yt4 != yt3:
                if np.round(np.abs(yt3 - yt4), 3) > np.max([yt3 * 1e-2, 0.01]):
                    print("d: " + str(d) + "h: " + str(h) + "yt3: " + str(yt3) + " yt4: " + str(yt4) + " unchi")
                    return np.zeros(19)
                else:
                    [yt, y1n] = [np.max([yt1, yt3, yt4]),
                                 np.array([y0n - 1, y0n + 1])[[yt1, yt3] == np.max([yt1, yt3])][0]]
            else:
                [yt, y1n] = [np.max([yt1, yt3]), np.array([y0n - 1, y0n + 1])[[yt1, yt3] == np.max([yt1, yt3])][0]]
    if w != 0:
        if z0n >= 0:
            [zt, z1n] = [np.ceil((Z[int(z0n + (w > 0)), y0n, x0n] - z0) / w * 100) / 100, int(z0n + np.sign(w))]
            # [zt, z1n] = [np.ceil((Z[int(z0n + (w > 0))] - z0) / w * 100) / 100, int(z0n + np.sign(w))]
        else:
            [zt, z1n] = [np.ceil((np.max([0, Z[0, y0n, x0n] + 200 * (z0n + (w > 0))]) - z0) / w * 100) / 100,
                         # [zt, z1n] = [np.ceil((np.max([0, Z[0] + 200 * (z0n + (w > 0))]) - z0) / w * 100) / 100,
                         int(z0n + np.sign(w))]
        if zt == 0:
            zt = float("inf")
    else:
        zt = float('inf')
        z1n = z0n
    if np.max(np.abs(np.array([u, v, w]) * np.array([xt, yt, zt]))) > 20000:
        xt, yt, zt = np.where(np.abs(np.array([u, v, w]) * np.array([xt, yt, zt])) > 20000,
                              np.inf, np.array([xt, yt, zt]))
    return [d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n, u, v, w, xt, yt, zt, x1n, y1n, z1n]


def func_hex_x(u, v, w, x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6,
               Z, d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n):
    # 5手前から1秒以上経過していなかった場合の千日手処理，x方向に細長い(y1, y2, y3とy4, y5, y6がほぼ同じ)三角形内部での振る舞い
    #   6--5--4
    #   |  0  |
    #   1--2--3
    x02, x21, x23, y02, y21, y23, x01, y01, x16, y16 = np.round([x2 - x0, x1 - x2, x3 - x2, y2 - y0, y1 - y2, y3 - y2,
                                                                 x1 - x0, y1 - y0, x6 - x1, y6 - y1], 4)
    [d21, d23, d16] = [x21 ** 2 + y21 ** 2, x23 ** 2 + y23 ** 2, x16 ** 2 + y16 ** 2]
    xt1 = ttb(u, v, d16, x01, x16, y01, y16)
    if xt1 > 0:
        xt2 = ttb(v, u, d16, y01, y16, x01, x16)
        if xt1 != xt2:
            if np.round(np.abs(xt1 - xt2), 3) > np.max([xt1 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "xt1: " + str(xt1) + " xt2: " + str(xt2) + " unchi")
                return np.zeros(19)
            else:
                [xt, x1n] = [np.max([xt1, xt2]), x0n - 1]
        else:
            [xt, x1n] = [xt1, x0n - 1]
    else:
        x03, x34, y03, y34 = np.round([x3 - x0, x4 - x3, y3 - y0, y4 - y3], 4)
        d34 = x34 ** 2 + y34 ** 2
        xt3 = ttb(u, v, d34, x03, x34, y03, y34)
        xt4 = ttb(v, u, d34, y03, y34, x03, x34)
        if xt4 != xt3:
            if np.round(np.abs(xt3 - xt4), 3) > np.max([xt3 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "xt3: " + str(xt3) + " xt4: " + str(xt4) + " unchi")
                return np.zeros(19)
            else:
                [xt, x1n] = [np.max([xt1, xt3, xt4]), np.array([x0n - 1, x0n + 1])[[xt1, xt3] == np.max([xt1, xt3])][0]]
        else:
            [xt, x1n] = [np.max([xt1, xt3]), np.array([x0n - 1, x0n + 1])[[xt1, xt3] == np.max([xt1, xt3])][0]]
    yt1 = ttb(u, v, d21, x02, x21, y02, y21)
    yt2 = ttb(u, v, d23, x02, x23, y02, y23)
    if np.min([yt1, yt2]) > 0:
        yt3 = ttb(v, u, d21, y02, y21, x02, x21)
        yt4 = ttb(v, u, d23, y02, y23, x02, x23)
        if (yt1 != yt3) | (yt2 != yt4):
            if np.round(np.abs(yt1 - yt3), 3) > np.max([yt1 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "yt1: " + str(yt1) + " yt3: " + str(yt3) + " unchi")
                return np.zeros(19)
            elif np.round(np.abs(yt2 - yt4), 3) > np.max([yt2 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "yt2: " + str(yt2) + " yt4: " + str(yt4) + " unchi")
                return np.zeros(19)
            else:
                [yt, y1n] = [np.min([np.max([yt1, yt2]), np.max([yt3, yt4])]), y0n - 1]
        else:
            [yt, y1n] = [np.min([yt1, yt2]), y0n - 1]
    else:
        x05, x54, x56, y05, y54, y56 = np.round([x5 - x0, x4 - x5, x6 - x5, y5 - y0, y4 - y5, y6 - y5], 4)
        d54, d56 = [x54 ** 2 + y54 ** 2, x56 ** 2 + y56 ** 2]
        yt5 = ttb(u, v, d54, x05, x54, y05, y54)
        yt6 = ttb(u, v, d56, x05, x56, y05, y56)
        yt7 = ttb(v, u, d54, y05, y54, x05, x54)
        yt8 = ttb(v, u, d56, y05, y56, x05, x56)
        if (yt5 != yt7) | (yt6 != yt8):
            if np.round(np.abs(yt5 - yt7), 3) > np.max([yt5 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "yt5: " + str(yt5) + " yt7: " + str(yt7) + " unchi")
                return np.zeros(19)
            elif np.round(np.abs(yt6 - yt8), 3) > np.max([yt6 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "yt6: " + str(yt6) + " yt8: " + str(yt8) + " unchi")
                return np.zeros(19)
            else:
                yt12, yt56 = np.min([yt1, yt2]), np.min([np.max([yt5, yt7]), np.max([yt6, yt8])])
                [yt, y1n] = [np.max([yt12, yt56]),
                             np.array([y0n - 1, y0n + 1])[[yt12, yt56] == np.max([yt12, yt56])][0]]
        else:
            yt12, yt56 = np.min([yt1, yt2]), np.min([yt5, yt6])
            [yt, y1n] = [np.max([yt12, yt56]), np.array([y0n - 1, y0n + 1])[[yt12, yt56] == np.max([yt12, yt56])][0]]
    if w != 0:
        if z0n >= 0:
            # [zt, z1n] = [np.ceil((Z[int(z0n + (w > 0)), y0n, x0n] - z0) / w * 100) / 100, int(z0n + np.sign(w))]
            [zt, z1n] = [np.ceil((Z[int(z0n + (w > 0))] - z0) / w * 100) / 100, int(z0n + np.sign(w))]
        else:
            # [zt, z1n] = [np.ceil((np.max([0, Z[0, y0n, x0n] + 200 * (z0n + (w > 0))]) - z0) / w * 100) / 100,
            [zt, z1n] = [np.ceil((np.max([0, Z[0] + 200 * (z0n + (w > 0))]) - z0) / w * 100) / 100,
                         int(z0n + np.sign(w))]
        if zt == 0:
            zt = float("inf")
    else:
        zt = float('inf')
        z1n = z0n
        # print("hex_x", d, h, np.array([xt, yt, zt]) - np.array(func(u, v, w, x1, x3, x6, x4, y1, y3, y6, y4, Z,
        #                                         d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n))[-6:-3])
    if np.max(np.abs(np.array([u, v, w]) * np.array([xt, yt, zt]))) > 20000:
        xt, yt, zt = np.where(np.abs(np.array([u, v, w]) * np.array([xt, yt, zt])) > 20000,
                              np.inf, np.array([xt, yt, zt]))
    return [d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n, u, v, w, xt, yt, zt, x1n, y1n, z1n]


def func_hex_y(u, v, w, x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6,
               Z, d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n):
    # 5手前から1秒以上経過していなかった場合の千日手処理，x方向に細長い(y1, y2, y3とy4, y5, y6がほぼ同じ)三角形内部での振る舞い
    #    5---4
    #   6  0  3
    #    1---2
    x06, x61, x65, y06, y61, y65, x01, y01, x12, y12 = np.round([x6 - x0, x1 - x6, x5 - x6, y6 - y0, y1 - y6, y5 - y6,
                                                                 x1 - x0, y1 - y0, x2 - x1, y2 - y1], 4)
    [d61, d65, d12] = [x61 ** 2 + y61 ** 2, x65 ** 2 + y65 ** 2, x12 ** 2 + y12 ** 2]
    xt1 = ttb(u, v, d61, x06, x61, y06, y61)
    xt2 = ttb(u, v, d65, x06, x65, y06, y65)
    if np.min([xt1, xt2]) > 0:
        xt3 = ttb(v, u, d61, y06, y61, x06, x61)
        xt4 = ttb(v, u, d65, y06, y65, x06, x65)
        if (xt1 != xt3) | (xt2 != xt4):
            if np.round(np.abs(xt1 - xt3), 3) > np.max([xt1 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "xt1: " + str(xt1) + " xt3: " + str(xt3) + " unchi")
                return np.zeros(19)
            elif np.round(np.abs(xt2 - xt4), 3) > np.max([xt2 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "xt2: " + str(xt2) + " xt4: " + str(xt4) + " unchi")
                return np.zeros(19)
            else:
                [xt, x1n] = [np.min([np.max([xt1, xt2]), np.max([xt3, xt4])]), x0n - 1]
        else:
            [xt, x1n] = [np.min([xt1, xt2]), x0n - 1]
    else:
        x03, x32, x34, y03, y32, y34 = np.round([x3 - x0, x2 - x3, x4 - x3, y3 - y0, y2 - y3, y4 - y3], 4)
        [d32, d34] = [x32 ** 2 + y32 ** 2, x34 ** 2 + y34 ** 2]
        xt5 = ttb(u, v, d32, x03, x32, y03, y32)
        xt6 = ttb(u, v, d34, x03, x34, y03, y34)
        xt7 = ttb(v, u, d32, y03, y32, x03, x32)
        xt8 = ttb(v, u, d34, y03, y34, x03, x34)
        if (xt5 != xt7) | (xt6 != xt8):
            if np.round(np.abs(xt5 - xt7), 3) > np.max([xt5 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "xt5: " + str(xt5) + " xt7: " + str(xt7) + " unchi")
                return np.zeros(19)
            elif np.round(np.abs(xt6 - xt8), 3) > np.max([xt6 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "xt6: " + str(xt6) + " xt8: " + str(xt8) + " unchi")
                return np.zeros(19)
            else:
                xt12, xt56 = np.min([xt1, xt2]), np.min([np.max([xt5, xt7]), np.max([xt6, xt8])])
                [xt, x1n] = [np.max([xt12, xt56]),
                             np.array([x0n - 1, x0n + 1])[[xt12, xt56] == np.max([xt12, xt56])][0]]
        else:
            xt12, xt56 = np.min([xt1, xt2]), np.min([xt5, xt6])
            [xt, x1n] = [np.max([xt12, xt56]), np.array([x0n - 1, x0n + 1])[[xt12, xt56] == np.max([xt12, xt56])][0]]
    yt1 = ttb(u, v, d12, x01, x12, y01, y12)
    if yt1 > 0:
        yt2 = ttb(v, u, d12, y01, y12, x01, x12)
        if yt1 != yt2:
            if np.round(np.abs(yt1 - yt2), 3) > np.max([yt1 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "yt1: " + str(yt1) + " yt2: " + str(yt2) + " unchi")
                return np.zeros(19)
            else:
                [yt, y1n] = [np.max([yt1, yt2]), y0n - 1]
        else:
            [yt, y1n] = [yt1, y0n - 1]
    else:
        x04, x45, y04, y45 = np.round([x4 - x0, x5 - x4, y4 - y0, y5 - y4], 4)
        d45 = x45 ** 2 + y45 ** 2
        yt3 = ttb(u, v, d45, x04, x45, y04, y45)
        yt4 = ttb(v, u, d45, y04, y45, x04, x45)
        if yt4 != yt3:
            if np.round(np.abs(yt3 - yt4), 3) > np.max([yt3 * 1e-2, 0.01]):
                print("d: " + str(d) + "h: " + str(h) + "yt3: " + str(yt3) + " yt4: " + str(yt4) + " unchi")
                return np.zeros(19)
            else:
                [yt, y1n] = [np.max([yt1, yt3, yt4]), np.array([y0n - 1, y0n + 1])[[yt1, yt3] == np.max([yt1, yt3])][0]]
        else:
            [yt, y1n] = [np.max([yt1, yt3]), np.array([y0n - 1, y0n + 1])[[yt1, yt3] == np.max([yt1, yt3])][0]]
    if w != 0:
        if z0n >= 0:
            # [zt, z1n] = [np.ceil((Z[int(z0n + (w > 0)), y0n, x0n] - z0) / w * 100) / 100, int(z0n + np.sign(w))]
            [zt, z1n] = [np.ceil((Z[int(z0n + (w > 0))] - z0) / w * 100) / 100, int(z0n + np.sign(w))]
        else:
            # [zt, z1n] = [np.ceil((np.max([0, Z[0, y0n, x0n] + 200 * (z0n + (w > 0))]) - z0) / w * 100) / 100,
            [zt, z1n] = [np.ceil((np.max([0, Z[0] + 200 * (z0n + (w > 0))]) - z0) / w * 100) / 100,
                         int(z0n + np.sign(w))]
        if zt == 0:
            zt = float("inf")
    else:
        zt = float('inf')
        z1n = z0n
    # print("hex_y", d, h, np.array([xt, yt, zt]) - np.array(func(u, v, w, x1, x2, x5, x4, y1, y2, y5, y4, Z,
    #                                             d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n))[-6:-3])
    if np.max(np.abs(np.array([u, v, w]) * np.array([xt, yt, zt]))) > 20000:
        xt, yt, zt = np.where(np.abs(np.array([u, v, w]) * np.array([xt, yt, zt])) > 20000,
                              np.inf, np.array([xt, yt, zt]))
    return [d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n, u, v, w, xt, yt, zt, x1n, y1n, z1n]
