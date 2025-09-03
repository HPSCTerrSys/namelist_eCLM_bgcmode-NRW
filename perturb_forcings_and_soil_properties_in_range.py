#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import netCDF4 as nc
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
import os
import json
import datetime
# from pandas.plotting import scatter_matrix
import pathlib

years = [2017]  # [2016, 2017]
months = [1]  # 1==January,...,12==December
num_ensemble = 4


# Helper functions
# Log normal to normal and vice versa
# for standard deviation and mean (formula from wikipedia)
# https://en.wikipedia.org/wiki/Log-normal_distribution
def ln_to_n(sd_ln, mean_ln):
    term = 1.0 + sd_ln * sd_ln / mean_ln / mean_ln
    return (np.sqrt(np.log(term)), np.log(mean_ln / np.sqrt(term)))


def n_to_ln(sd_n, mean_n):
    return ((np.exp(sd_n * sd_n) - 1.0) * np.exp(2.0 * mean_n + sd_n * sd_n),
            np.exp(mean_n + sd_n * sd_n / 2.0))


# Helper function to serialize / deserialize random state with json


def rnd_state_serialize():
    tmp_state = np.random.get_state()
    save_state = ()
    for i in tmp_state:
        if type(i) is np.ndarray:
            save_state = save_state + (i.tolist(), )
        else:
            save_state = save_state + (i, )
    json.dump(save_state, open("rnd_state.json", "w"))


def rnd_state_deserialize():
    tmp_state = json.load(open("rnd_state.json", "r"))
    load_state = ()
    for i in tmp_state:
        if type(i) is list:
            load_state = load_state + (np.array(i), )
        else:
            load_state = load_state + (i, )
    np.random.set_state(load_state)


# # Correlation / scatter matrix plot formatting from
# # https://stackoverflow.com/a/50690729
# def corr_dot(*args, **kwargs):
#    corr_r = args[0].corr(args[1], "pearson")
#    corr_text = f"{corr_r:2.3f}"
#    ax = plt.gca()
#    ax.set_axis_off()
#    marker_size = abs(corr_r) * 17500
#    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
#               vmin=-1, vmax=1, transform=ax.transAxes)
#    font_size = abs(corr_r) * 40 + 5
#    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
#                ha="center", va="center", fontsize=font_size)

# def plot_corr(perturbations, pname="Plot.png"):
#     fig = plt.figure()
#     sns.set(style="white", font_scale=1.6)
#     df = pd.DataFrame(perturbations,
#                       columns=["Precip", "SW", "LW", "Temp2m"])
#     g = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
#     g.map_lower(sns.scatterplot)
#     g.map_diag(sns.distplot, kde_kws={"color": "black"})
#     g.map_upper(corr_dot)
#     plt.savefig(pname, format="png", dpi=300, bbox_inches="tight")

# # Alternative way to plot / plots symmetric scatter matrix and just adds
# # text on top
# def plot_corr(perturbations):
#     fig = plt.figure()
#     df = pd.DataFrame(perturbations,
#                       columns=["Precip", "SW", "LW", "Temp2m"])
#     axes = scatter_matrix(df, alpha=0.2, figsize=(4, 4), diagonal="kde")
#     corr = df.corr().values
#     for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
#         axes[i, j].annotate("%.3f" % corr[i, j], (0.5, 0.5),
#                             xycoords="axes fraction",
#                             ha="center",
#                             va="center",
#                             fontweight="bold")

#     plt.savefig("Plot_Perturbed_Forcing_Scatter_Matrix.png",
#                 format="png",
#                 dpi=300,
#                 bbox_inches="tight")
#     plt.show()

# # Helper function to calculate downward longwave radiation flux
# # from pressure, relative humidity and temperature.
# # Same function as done in clm 3.5 if no FLDS is present in atm forcing
# # files.
# # see clm3.5/src/main/atmdrvMod.f90 line 939ff
# # # No need if FLDS is there

# def tdc(t):
#     return (np.minimum(50.0, np.maximum(-50.0, t - 273.15)))  #

# def esatw(t):
#     return (100.0 * (6.107799961 + t *
#                      (4.436518521e-1 + t *
#                       (1.428945805e-2 + t *
#                        (2.650648471e-4 + t *
#                         (3.031240396e-6 + t *
#                          (2.034080948e-8 + t * 6.136820929e-11)))))))

# def esati(t):
#     return (100.0 * (6.109177956 + t *
#                      (5.034698970e-1 + t *
#                       (1.886013408e-2 + t *
#                        (4.176223716e-4 + t *
#                         (5.824720280e-6 + t *
#                          (4.838803174e-8 + t * 1.838826904e-10)))))))

# def clm3_5_flds_calc(psrf, rh, t):
#     vp = np.where(t > 273.15, rh / 100.0 * esatw(tdc(t)),
#                   rh / 100.0 * esati(tdc(t)))
#     # Could just use this "vp" - but in clm 3.5 if RH is given it is
#     # converted like this.
#     q = 0.622 * vp / (psrf - 0.378 * vp)
#     e = psrf * q / (0.622 + 0.378 * q)
#     ea = 0.7 + 5.95e-5 * 0.01 * e * np.exp(1500.0 / t)
#     return (ea * 5.67e-8 * t**4)


# Helper function - copy attributes and dimensions
def copy_attr_dim(src, dst, usr=None):
    # copy attributes
    for name in src.ncattrs():
        dst.setncattr("original_attribute_" + name, src.getncattr(name))
    # copy dimensions
    for name, dimension in src.dimensions.items():
        dst.createDimension(name, len(dimension))
    # Additional attribute
    if usr is None:
        usr = os.getlogin()
    dst.setncattr("perturbed_by", usr)
    dst.setncattr("perturbed_on_date",
                  datetime.datetime.today().strftime("%d.%m.%y"))


def perturb_soil_properties(iensemble=0,
                            sorigdir="./input_clm/",
                            snamedir="./input_clm/"):
    """
    Perturb soil properties in surface data NetCDF file.

    Format of NetCDF file: ``surfdata_*.nc``.

    Perturbed NetCDF files get the index of the ensemble member (``i`` starting
    with ``1``) appended: ``surfdata_*.nc_iiiii.nc*

    # Sand and clay content were perturbed with random noise drawn from
    # spatially uniform distribution (±10 %). In order to avoid un-physical
    # values of the soil parameters, the sum of the sand and clay content
    # were constrained to have a value not larger than 100 %.

    Parameters
    ----------
    iensemble : int
       Ensemble member to generate.

    sorigdir : str
       Path to original surfdata file. Default ``./input_clm/``.

    snamedir : str
       Path for perturbed surfdata files. Default ``./input_clm/``.

    """
    sname = (
        snamedir +
        "surfdata_300x300_NRW_hist_78pfts_CMIP6_simyr2000_c190619" +
        "_" + str(iensemble + 1).zfill(5) + ".nc")
    sorig = (
        sorigdir +
        "surfdata_300x300_NRW_hist_78pfts_CMIP6_simyr2000_c190619.nc")

    with nc.Dataset(sorig, "r") as src, nc.Dataset(sname, "w") as dst:
        # Copy attributes
        copy_attr_dim(src, dst)
        # dimension of perturbed fields
        dim_lvl = src.dimensions["nlevsoi"].size
        dim_lat = src.dimensions["lsmlat"].size
        dim_lon = src.dimensions["lsmlon"].size
        dim_types = 3

        # Perturbations:
        rnd_type_cell = np.random.uniform(low=-20.0,
                                          high=20.0,
                                          size=dim_lat * dim_lon *
                                          dim_types).reshape(
                                              dim_types, dim_lat * dim_lon)
        rnd = np.zeros((dim_types, dim_lvl, dim_lat * dim_lon))
        for t in range(dim_types):
            for c in range(dim_lat * dim_lon):
                rnd[t, :, c] = rnd_type_cell[t, c]
        rnd = rnd.reshape((dim_types, dim_lvl, dim_lat, dim_lon))

        # Keep percentages normalized (sum to 100)
        pct = np.array([
            src.variables["PCT_SAND"][:] + rnd[0],
            src.variables["PCT_CLAY"][:] + rnd[1]
        ])
        # src.variables["ORGANIC"][:] + rnd[2]])

        pct_om = np.array([src.variables["ORGANIC"][:] + rnd[2]])

        # Minimum and maximum values for sand and clay percentages
        pct_sand_min = 1.0
        pct_sand_max = 99.0
        pct_clay_min = 1.0
        pct_clay_max = 99.0

        # Keep in range between 0 and 99 percent
        for ll in range(dim_lvl):
            for la in range(dim_lat):
                for lo in range(dim_lon):
                    #                   for t in range(dim_types):
                    if pct[0, ll, la, lo] > pct_sand_max:
                        pct[0, ll, la, lo] = pct_sand_max
                    if pct[1, ll, la, lo] > pct_clay_max:
                        pct[1, ll, la, lo] = pct_clay_max
                    if pct[0, ll, la, lo] < pct_sand_min:
                        pct[0, ll, la, lo] = pct_sand_min
                    if pct[1, ll, la, lo] < pct_clay_min:
                        pct[1, ll, la, lo] = pct_clay_min
        # Keep OM in range
                    if pct_om[:, ll, la, lo] > 120.0:
                        pct_om[:, ll, la, lo] = 120.0
                    if pct_om[:, ll, la, lo] < 0.0:
                        pct_om[:, ll, la, lo] = 0.0

        # Keep percentages normalized (sum to 100)
        for ll in range(dim_lvl):
            for la in range(dim_lat):
                for lo in range(dim_lon):
                    old_sum = np.sum(pct[:, ll, la, lo])
                    for t in range(dim_types):
                        if old_sum > 100.0:
                            pct[:, ll, la,
                                lo] = 100.0 * pct[:, ll, la, lo] / old_sum
                        elif old_sum <= 0.0:
                            raise RuntimeError("Sum of percentages of clay and sand has to be larger than zero. Consider adapting pct_sand_min and pct_clay_min.")

        # Copy non-perturbed variables:
        for name, var in src.variables.items():
            if name != "PCT_SAND" and name != "PCT_CLAY" and name != "ORGANIC":
                dst.createVariable(name, var.datatype, var.dimensions)
                dst[name].setncatts(src[name].__dict__)
                dst[name][:] = src[name][:]

        # Add perturbations
        pct_sand = dst.createVariable("PCT_SAND",
                                      datatype=np.float64,
                                      dimensions=(
                                          "nlevsoi",
                                          "lsmlat",
                                          "lsmlon",
                                      ),
                                      fill_value=1.e+30)
        pct_sand.setncatts({
            'long_name': u"percent sand",
            'units': u"unitless"
        })
        dst.variables["PCT_SAND"][:] = pct[0].reshape(
            dst.variables["PCT_SAND"].shape)

        pct_clay = dst.createVariable("PCT_CLAY",
                                      datatype=np.float64,
                                      dimensions=(
                                          "nlevsoi",
                                          "lsmlat",
                                          "lsmlon",
                                      ),
                                      fill_value=1.e+30)
        pct_clay.setncatts({
            'long_name': u"percent clay",
            'units': u"unitless"
        })
        dst.variables["PCT_CLAY"][:] = pct[1].reshape(
            dst.variables["PCT_CLAY"].shape)

        om = dst.createVariable("ORGANIC",
                                datatype=np.float64,
                                dimensions=(
                                    "nlevsoi",
                                    "lsmlat",
                                    "lsmlon",
                                ),
                                fill_value=1.e+30)
        om.setncatts({
            'long_name': u"organic matter density at soil levels",
            'units': u"kg/m3 (assumed carbon content 0.58 gC per gOM)"
        })
        dst.variables["ORGANIC"][:] = pct_om.reshape(
            dst.variables["ORGANIC"].shape)


def perturb_nc_file(year=2006,
                    month=1,
                    iensemble=0,
                    fdir="./forcings/",
                    outdir="./forcings/"):
    """
    Perturb a single month forcing NetCDF file.

    Format of NetCDF file: ``yyyy-mm.nc``.

    Perturbed NetCDF files will be in subfolders of the form ``real_iiiii/``,
    where ``i`` is denotes the ensemble member starting with ``1``.

    Parameters
    ----------
    Year : int
        Year in name of forcing files.

    month : int
        Month in name of forcing files.

    iensemble : int
       Ensemble member to generate.

    fdir : str
       Path to original forcing file. Default: ``./forcings/``.

    outdir : str
       Path for output subdirectories for perturbed ensemble of forcing files.
       Default: ``./forcings/``.

    """
    # standard deviation and mean of perturbations
    sd = [ln_to_n(0.5, 1.0)[0], ln_to_n(0.3, 1.0)[0], 20.0, 1.0]
    mean = [ln_to_n(0.5, 1.0)[1], ln_to_n(0.3, 1.0)[1], 0.0, 0.0]

    # Correlation matrix based on:
    # Reichle et al. [2007] https://doi.org/10.1029/2006JD008033
    # Han et al. [2014] https://doi:10.1002/2013WR014586
    # Precipitation | ShortWave Radiation |
    # LongWave Radiation | Air Temperature at 2m
    correl = np.array([[1.0, -0.8, 0.5, 0.0], [-0.8, 1.0, -0.5, 0.4],
                       [0.5, -0.5, 1.0, 0.4], [0.0, 0.4, 0.4, 1.0]])

    # fname = ("../atmforcing/forcings_orig/" + str(year) + "-" +
    #          str(month).zfill(2) + ".nc")
    fname = (fdir + str(year) + "-" + str(month).zfill(2) + ".nc")
    outname = pathlib.Path(outdir + "real_" + str(iensemble + 1).zfill(5) +
                           "/" + str(year) + "-" + str(month).zfill(2) + ".nc")

    # Create subdirectories for forcing perturbation
    outname.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(fname, "r") as src, nc.Dataset(outname, "w") as dst:

        copy_attr_dim(src, dst)

        dim_time = src.dimensions["time"].size
        dim_lat = src.dimensions["lat"].size
        dim_lon = src.dimensions["lon"].size

        # Perturbations:
        # Use built-in multivariate, correlated pseudo-number generator
        # directly
        # instead of Cholesky decomposition and matrix-matrix multiplication
        # manually
        rnd = np.random.multivariate_normal(np.zeros_like(mean), correl,
                                            dim_time * dim_lat * dim_lon)

        # Precipitation and ShortWave are log normal distributed perturbations
        # therefore exponential from normal distributed pseudo-random number.

        #        if plotting:
        #            plot_corr(rnd, "Plot_Perturbation_RV_Scatter_Matrix.png")

        perturbations = np.zeros_like(rnd)
        perturbations[:, 0] = np.exp(mean[0] + sd[0] * rnd[:, 0])
        perturbations[:, 1] = np.exp(mean[1] + sd[1] * rnd[:, 1])
        perturbations[:, 2] = mean[2] + rnd[:, 2] * sd[2]
        perturbations[:, 3] = mean[3] + rnd[:, 3] * sd[3]

        # if plotting:
        #     plot_corr(perturbations, "Plot_Perturbation_Scatter_Matrix.png")

        # After generating all random variables
        # save state of random number generator to file
        if not force_seed:
            rnd_state_serialize()

        # netCDF variables
        # Copy non-perturbed variables:
        for name, var in src.variables.items():
            if (name != "TBOT" and name != "PRECTmms" and name != "FSDS"
                    and name != "FLDS"):
                dst.createVariable(name, var.datatype, var.dimensions)
                dst[name].setncatts(src[name].__dict__)
                dst[name][:] = src[name][:]
        # Add / multiply perturbations
        tbot = dst.createVariable("TBOT",
                                  datatype=np.float64,
                                  dimensions=(
                                      "time",
                                      "lat",
                                      "lon",
                                  ),
                                  fill_value=9.96921e+36)
        tbot.setncatts({
            "height": u"2",
            "units": u"K",
            "missing_value": -9.e+33
        })
        dst.variables["TBOT"][:, :, :] = (
            src.variables["TBOT"][:, :, :] +
            perturbations[:, 3].reshape(src.variables["TBOT"][:, :, :].shape))

        prectmms = dst.createVariable("PRECTmms",
                                      datatype=np.float64,
                                      dimensions=(
                                          "time",
                                          "lat",
                                          "lon",
                                      ),
                                      fill_value=-9.e+33)
        prectmms.setncatts({"units": u"mm/s", "missing_value": -9.e+33})
        dst.variables["PRECTmms"][:] = (
            src.variables["PRECTmms"][:] * perturbations[:, 0].reshape(
                src.variables["PRECTmms"][:, :, :].shape))

        fsds = dst.createVariable("FSDS",
                                  datatype=np.float64,
                                  dimensions=(
                                      "time",
                                      "lat",
                                      "lon",
                                  ),
                                  fill_value=-9.e+33)
        fsds.setncatts({"missing_value": -9.e+33})
        dst.variables["FSDS"][:] = (
            src.variables["FSDS"][:] *
            perturbations[:, 1].reshape(src.variables["FSDS"][:, :, :].shape))

        # flds = dst.createVariable("FLDS",
        #                           datatype=np.float64,
        #                           dimensions=(
        #                               "time",
        #                               "lat",
        #                               "lon",
        #                           ),
        #                           fill_value=-9.e+33)
        # flds.setncatts({"missing_value": -9.e+33})
        # dst.variables["FLDS"][:] = (
        #     src.variables["FLDS"][:] *
        #     perturbations[:, 2].reshape(
        #         src.variables["FLDS"][:, :, :].shape))


#       dst.variables["FLDS"][:] = (clm3_5_flds_calc(src.variables["PSRF"][:],
#                                                    src.variables["RH"][:],
#                                                   src.variables["TBOT"][:]) +
#                                   perturbations[:, 2].reshape(
#                                   src.variables["FSDS"][:, :, :].shape))

# Settings / parameters
# plotting = False
rnd_state_file = "rnd_state.json"
force_seed = True
# Either seed random number generator or continue with existing state
if not os.path.isfile(rnd_state_file) or force_seed:
    # TODO: Move to np.random Generator instances
    # https://numpy.org/doc/stable/reference/random/index.html
    # np.random.seed(42)
    np.random.seed(
        np.square(np.sum([ord(c) for c in os.getenv("USER")]))
    )
else:
    rnd_state_deserialize()
#    rnd_state = pickle.load(open(rnd_state_file, "rb"))
#    np.random.set_state(rnd_state)

for ens in range(num_ensemble):
    perturb_soil_properties(ens)
    for y in years:
        print("Done with year " + str(y) + " ensemble " + str(ens + 1))
        for m in months:
            perturb_nc_file(y, m, ens)
