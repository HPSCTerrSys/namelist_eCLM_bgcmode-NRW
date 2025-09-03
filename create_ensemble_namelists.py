#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
# import pathlib

import f90nml
from lxml import etree
from io import BytesIO


def adapt_modelio(mod, blddir, rundir, iens):
    nml = f90nml.read(mod + "_modelio.nml")
    nml.indent = 1

    nml["modelio"]["logfile"] = nml["modelio"]["logfile"].replace(
        ".log", "_" + str(iens).zfill(4) + ".log")

    nml.write(mod + "_modelio.nml_" + str(iens).zfill(4))

    # with open(mod + "_modelio.nml_" + str(iens).zfill(4), "w+") as nl:
    #     nl.write(nl_string)


def adapt_cpl_modelio(blddir, rundir):
    nml = f90nml.read("cpl_modelio.nml")
    nml.indent = 1
    nml.write("cpl_modelio.nml_adapted")


def adapt_mosart_in(iens):
    nml = f90nml.read("mosart_in")
    nml.indent = 1
    nml.write("mosart_in_" + str(iens).zfill(4))


def adapt_datm_in(syear, eyear, iens, domname=None):
    nml = f90nml.read("datm_in")
    nml.indent = 1

    if not domname is None:
        nml["shr_strdata_nml"]["domainfile"] = "./input_clm/" + domname

    for i, streamfile in enumerate(nml["shr_strdata_nml"]["streams"]):
        nml["shr_strdata_nml"]["streams"][i] = streamfile.replace(
            ".txt", "_" + str(iens).zfill(4) + ".txt")

    nml.write("datm_in_" + str(iens).zfill(4))


def adapt_drv_flds_in():
    nml = f90nml.read("drv_flds_in")
    nml.indent = 1
    nml.write("drv_flds_in_adapted")


def adapt_drv_in(rundir, syear, eyear, prefix=None):
    nml = f90nml.read("drv_in")
    nml.indent = 1

    if prefix is not None:
        nml["seq_infodata_inparm"]["case_name"] = prefix

    nml["seq_timemgr_inparm"]["start_ymd"] = syear * 10000 + 101
    nml["seq_timemgr_inparm"]["stop_ymd"] = eyear * 10000 + 131

    nml.write("drv_in_adapted")


def adapt_lnd_in(syear, eyear, iens, prefix="nrw_300x300"):
    nml = f90nml.read("lnd_in")
    nml.indent = 1

    nml["clm_inparm"]["fsurdat"] = nml["clm_inparm"]["fsurdat"].replace(
        ".nc", "_" + str(iens).zfill(5) + ".nc")

    nml.write("lnd_in_" + str(iens).zfill(4))


def adapt_stream_files(iens):

    # Must be called after datm update

    # Read original stream files and new stream files
    nml_datm = f90nml.read("datm_in")
    nml_datm_ens = f90nml.read("datm_in_" + str(iens).zfill(4))

    for i, streamfile in enumerate(nml_datm["shr_strdata_nml"]["streams"]):
        tree = etree.parse(streamfile.split()[0])
        root = tree.getroot()

        # Extract element "filePath" with parent element "fieldInfo"
        for filePath in root.iter("filePath"):
            # Check parent
            if filePath.getparent().tag == "fieldInfo":
                # print(filePath.getparent().tag, filePath.tag)
                if filePath.text.find("./forcings") > -1:
                    filePath.text = filePath.text.replace(
                        "./forcings", "./forcings/real_" + str(iens).zfill(5))

        # Write XML streamfile
        # --------------------

        # Use buffer first
        fbuffer = BytesIO()
        tree.write(fbuffer, xml_declaration=True, encoding="ASCII")
        fstr = fbuffer.getvalue().decode("ASCII")

        # Replace XML declaration to match original
        fstr = fstr.replace("version='1.0'", "version=\"1.0\"")
        fstr = fstr.replace(" encoding='ASCII'", "")

        # Write to file
        with open(nml_datm_ens["shr_strdata_nml"]["streams"][i].split()[0],
                  "w+") as f:
            f.write(fstr)


def create_ensemble_namelists(blddir,
                              rundir,
                              syear,
                              eyear,
                              iens,
                              prefix="nrw_300x300"):
    for m in ["atm", "esp", "glc", "ice", "lnd", "ocn", "rof", "wav"]:
        adapt_modelio(m, blddir, rundir, iens)

    adapt_datm_in(syear, eyear, iens)

    adapt_lnd_in(syear, eyear, iens, prefix)

    adapt_mosart_in(iens)

    adapt_stream_files(iens)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        year_start = int(sys.argv[1])
    else:
        year_start = 2017

    if len(sys.argv) > 2:
        prefix = str(sys.argv[2])
    else:
        prefix = "nrw_300x300"

    year_end = 2017

    num_ensemble = 4

    # Get run directory as current directory and build directory from the
    # tsmp-pdaf symbolic link.
    cwd = os.getcwd()
    sl = os.readlink("tsmp-pdaf")
    b_dir = sl.split("tsmp-pdaf")[0]
    # sl = os.readlink("tsmp-pdaf")
    # b_dir = sl.split("tsmp-pdaf")[0]
    r_dir = cwd

    adapt_drv_in(r_dir, year_start, year_end, prefix)

    adapt_drv_flds_in()

    adapt_cpl_modelio(b_dir, r_dir)

    for ens in range(1, num_ensemble + 1):
        if prefix is None:
            create_ensemble_namelists(b_dir, r_dir, year_start, year_end, ens)
        else:
            create_ensemble_namelists(b_dir, r_dir, year_start, year_end, ens,
                                      prefix)
        sys.stdout.write("\r[%s] " %
                         ("Done with ensemble member: " + str(ens)))
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
