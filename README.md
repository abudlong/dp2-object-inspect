# dp2-object-inspect

A Jupyter notebook for inspecting a single transient or variable source in Rubin/LSST
Data Release Processing (DRP) outputs using the LSST Science Pipelines Butler. You give it
a sky position, and it collects every visit, coadd, difference-imaging product and DIA
catalog entry at that position, then plots them.

## What it produces

| Section | Output |
|---|---|
| 1. Visit images | Table of every `visit_image` (visit + detector) whose footprint overlaps the position |
| 2. Coadds | Table of overlapping `deep_coadd` and `template_coadd` patches |
| 3. DIA catalogs | Positional match to `dia_object`, its `dia_source` detections, reliability cut |
| 4. Cutouts | Visit image / template / difference cutouts; PSF model scaled to AP, Calib and PSF flux |
| 4b. PSF | Visit PSF model, detector map of PSF stars (`calib_psf_used` / `reserved`), star − model stamps |
| 4c. Cross sections | Smoothed horizontal/vertical cuts through visit image, template and difference |
| 4d. PSF fit | Difference image vs fitted PSF model: images, smoothed cuts, radial profile |
| 5. Light curve | Interactive (plotly) flux vs date for AP / Calib / PSF fluxes, per band |
| 6. Template inputs | Visits/detectors that built the template at the position: interactive footprint map with overlap density, epochs vs light curve, per-input thumbnails |

## Requirements

- An LSST Science Pipelines environment (tested against the `rubin-env` stack used for
  DP2 processing, weekly/v30-era), e.g. the Rubin Science Platform or USDF.
- Read access to a Butler repository containing DRP outputs. The notebook was written
  against the internal `dp2_prep` repository; you need your own access rights to it.
- Extra Python packages not always present in the stack: see `requirements.txt`
  (`plotly`, `scipy`).

## Usage

1. Open `notebooks/dp2_object_inspect.ipynb` with the LSST kernel.
2. Edit the configuration cell at the top:

   ```python
   REPO, RUN = "dp2_prep", "LSSTCam/runs/DRP/DP2"   # Butler repo + (chained) collection
   OBJ_ID = 170028500566147171                       # external ID, used as a label only
   ra, dec = 149.879551, 3.222127                    # target position [deg]
   ```

   Other options: `CUTOUT_PX`, `MATCH_RADIUS_ARCSEC`, `INCLUDE_ALL_MATCHES`,
   `RELIABILITY_MIN`, `CROSS_WIDTH`, `N_PSF_STARS`, `VISIT`/`DETECTOR` (to force the
   cutout visit), plus `TEMPLATE_BAND`, `MAX_THUMBS` and `SMOOTH_SIGMA` in later cells.
3. Run all cells.

Use a **chained collection** that covers all pipeline stages. A single stage run (for
example `.../stage4/<timestamp>`) does not contain `visit_image` or the coadds.

## Notes and caveats

- **Prompt Processing vs DRP IDs.** Prompt Processing / alert-broker (e.g. Lasair)
  `diaObjectId`s are not the same as DRP `diaObjectId`s. The notebook matches on position
  and lists every DRP object within `MATCH_RADIUS_ARCSEC`, with separation, number of
  detections and reliability. Confirm a match by comparing detections visit by visit.
- **"Calib" flux.** The DPDD `dia_source` schema has no `calibFlux`. If it is missing, the
  notebook uses `scienceFlux` (PSF flux on the direct image) and says so. That is a total
  flux, not a difference flux, so the light curves plot it on a secondary y axis.
- **Reliability.** Detections with `reliability < RELIABILITY_MIN` are dropped. If none
  pass, the notebook warns and continues with all detections, so the plots still render.
- **Template self-subtraction.** Section 6 flags detections taken within a day of a
  template input visit. Their difference fluxes are partly cancelled by the template.
- **Dataset type names.** Current names (`visit_image`, `difference_image`,
  `template_detector`, `deep_coadd`, `template_coadd`, `dia_object`, `dia_source`,
  `source`) are tried first, then older Gen3 names as fallbacks.
- **Cell-based coadds.** Section 6 reads `coaddInputs` from `template_coadd`. A cell-based
  template may not carry that component; the cell stops with a message if so.

## Repository layout

```
notebooks/dp2_object_inspect.ipynb   # the notebook (generated, outputs stripped)
scripts/dp2_object_inspect.py        # same code as a percent-format script (source of truth)
tools/py_to_ipynb.py                 # rebuild the notebook from the script
jupytext.toml                        # optional: pair the two with jupytext
```

The `.py` file is the version to edit and review, because it diffs cleanly. After changing it,
regenerate the notebook:

```bash
python tools/py_to_ipynb.py
```

Or, with [jupytext](https://jupytext.readthedocs.io/) installed, edit either file and run
`jupytext --sync scripts/dp2_object_inspect.py`.

Please commit notebooks **without outputs**: images and catalog values from
embargoed or pre-release data should not end up in the repository.
