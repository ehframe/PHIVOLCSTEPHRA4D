import netCDF4 as nc
import numpy as np
from wrf import getvar, ALL_TIMES
dir = "F:/PAGASA_NMS_wrfout/20250407_0000/"

# Creating a simple test list with three timesteps
wrflist = [nc.Dataset(dir + "wrfout_d02_2025-04-08_05_00_00"),
           nc.Dataset(dir + "wrfout_d02_2025-04-08_06_00_00"),
           nc.Dataset(dir + "wrfout_d02_2025-04-08_07_00_00"),
           nc.Dataset(dir + "wrfout_d02_2025-04-08_08_00_00"),
           nc.Dataset(dir + "wrfout_d02_2025-04-08_09_00_00"),
           nc.Dataset(dir + "wrfout_d02_2025-04-08_10_00_00"),
           nc.Dataset(dir + "wrfout_d02_2025-04-08_11_00_00"),
           nc.Dataset(dir + "wrfout_d02_2025-04-08_12_00_00")]

# Extract the variables for all times
P_cat = getvar(wrflist, "P", timeidx=ALL_TIMES, method="cat")
PH_cat = getvar(wrflist, "PH", timeidx=ALL_TIMES, method="cat")
PB_cat = getvar(wrflist, "PB", timeidx=ALL_TIMES, method="cat")
PHB_cat = getvar(wrflist, "PHB", timeidx=ALL_TIMES, method="cat")
HGT_cat = getvar(wrflist, "HGT", timeidx=ALL_TIMES, method="cat")
U_cat = getvar(wrflist, "U", timeidx=ALL_TIMES, method="cat")
V_cat = getvar(wrflist, "V", timeidx=ALL_TIMES, method="cat")
W_cat = getvar(wrflist, "W", timeidx=ALL_TIMES, method="cat")
XLONG_cat = getvar(wrflist, "XLONG", timeidx=ALL_TIMES, method="cat")
XLAT_cat = getvar(wrflist, "XLAT", timeidx=ALL_TIMES, method="cat")
QVAPOR_cat = getvar(wrflist, "QVAPOR", timeidx=ALL_TIMES, method="cat")
T_cat = getvar(wrflist, "T", timeidx=ALL_TIMES, method="cat")
XTIME_cat = getvar(wrflist, "XTIME", timeidx=ALL_TIMES, method="cat")

# Export data into new nc
ext_nc = nc.Dataset(dir + "extracted3.nc", "w",
                         format="NETCDF4_CLASSIC")
ext_nc.createDimension("time", 8)
ext_nc.createDimension("bottom_top", 49)
ext_nc.createDimension("south_north", 592)
ext_nc.createDimension("west_east", 360)
ext_nc.createDimension("U_west_east", 361)
ext_nc.createDimension("V_south_north", 593)
ext_nc.createDimension("W_bottom_top", 50)


ext_nc_P = ext_nc.createVariable("P", np.dtype("float32").char, ("time", "bottom_top", "south_north", "west_east"))
ext_nc_P.long_name = "Perturbation pressure"
ext_nc_P.units = "Pa"

ext_nc_PH = ext_nc.createVariable("PH", np.dtype("float32").char, ("time", "W_bottom_top", "south_north", "west_east"))
ext_nc_PH.long_name = "Perturbation geopotential height"
ext_nc_PH.units = "m2 s-2"

ext_nc_PB = ext_nc.createVariable("PB", np.dtype("float32").char, ("time", "bottom_top", "south_north", "west_east"))
ext_nc_PB.long_name = "Base state pressure"
ext_nc_PB.units = "Pa"

ext_nc_PHB = ext_nc.createVariable("PHB", np.dtype("float32").char, ("time", "W_bottom_top", "south_north", "west_east"))
ext_nc_PHB.long_name = "Base geopotential height"
ext_nc_PHB.units = "m2 s-2"

ext_nc_HGT = ext_nc.createVariable("HGT", np.dtype("float32").char, ("time", "south_north", "west_east"))
ext_nc_HGT.long_name = "Terrain height"
ext_nc_HGT.units = "m"

ext_nc_U = ext_nc.createVariable("U", np.dtype("float32").char, ("time", "bottom_top", "south_north", "U_west_east"))
ext_nc_U.long_name = "Zonal wind component"
ext_nc_U.units = "m s-1"

ext_nc_V = ext_nc.createVariable("V", np.dtype("float32").char, ("time", "bottom_top", "V_south_north", "west_east"))
ext_nc_V.long_name = "Meridional wind component"
ext_nc_V.units = "m s-1"

ext_nc_W = ext_nc.createVariable("W", np.dtype("float32").char, ("time", "W_bottom_top", "south_north", "west_east"))
ext_nc_W.long_name = "Vertical wind component"
ext_nc_W.units = "m s-1"

ext_nc_XLONG = ext_nc.createVariable("XLONG", np.dtype("float32").char, ("time", "south_north", "west_east"))
ext_nc_XLONG.long_name = "Longitude"
ext_nc_XLONG.units = "degree east"

ext_nc_XLAT = ext_nc.createVariable("XLAT", np.dtype("float32").char, ("time", "south_north", "west_east"))
ext_nc_XLAT.long_name = "Latitude"
ext_nc_XLAT.units = "degree nort"

ext_nc_QVAPOR = ext_nc.createVariable("QVAPOR", np.dtype("float32").char, ("time", "bottom_top", "south_north", "west_east"))
ext_nc_QVAPOR.long_name = "Water vapor mixing ratio"
ext_nc_QVAPOR.units = "kg kg-1"

ext_nc_T = ext_nc.createVariable("T", np.dtype("float32").char, ("time", "bottom_top", "south_north", "west_east"))
ext_nc_T.long_name = "Water vapor mixing ratio"
ext_nc_T.units = "K"

ext_nc_XTIME = ext_nc.createVariable("XTIME", np.dtype("float32").char, ("time"))
ext_nc_XTIME.long_name = "Delta time"
ext_nc_XTIME.units = "minutes since 2025/04/08 5:00"

ext_nc_P[:] = P_cat
ext_nc_PH[:] = PH_cat
ext_nc_PB[:] = PB_cat
ext_nc_PHB[:] = PHB_cat
ext_nc_HGT[:] = HGT_cat
ext_nc_U[:] = U_cat
ext_nc_V[:] = V_cat
ext_nc_W[:] = W_cat
ext_nc_XLONG[:] = XLONG_cat
ext_nc_XLAT[:] = XLAT_cat
ext_nc_QVAPOR[:] = QVAPOR_cat
ext_nc_T[:] = T_cat
ext_nc_XTIME[:] = XTIME_cat

ext_nc.close()