# eCLM namelist NRW

eCLM namelist for NRW test in biogeochemistry (BGC) mode. Used in the
FallSchool 2025 (Day 1/2).


## Changes compared to `master`
- jobscript `run-eclm-job.sh`
  - sets the environment file and executable explicitly, while for
    `master` symlinks named `loadenvs` and `tsmp-pdaf` have to be
    defined.
  - makes directories `logs` and `timing/checkpoints`
- `drv_in` / jobscript: eCLM uses 96 tasks instead of 32 in `master`
- `lnd_in`: `hist_mfilt = 365` instead of 10 for `master`
- `forcings`: symlinks to all forcings throughout 2017
- `input_clm`: symlinks changed to Day1_2 paths
