# %% [markdown]
# # DP2 object inspector
# Visits, coadds, cutouts, PSF models, template inputs and light curve for one sky position.
#
# Run inside an LSST Science Pipelines environment (RSP / USDF) with access to the
# Butler repo named in `REPO`. Edit the configuration cell below, then run all cells.

# %%
# ---- Configuration -------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
import astropy.units as u

import lsst.geom as geom
from lsst.daf.butler import Butler

REPO, RUN = "dp2_prep", "LSSTCam/runs/DRP/DP2"   # chained collection (all stages)
OBJ_ID = 170028500566147171                 # Lasair / Prompt Processing ID (label only)
ra, dec = 149.879551, 3.222127

COLLECTIONS = [RUN]

CUTOUT_PX = 41                 # cutout size (pixels)
MATCH_RADIUS_ARCSEC = 1.0      # dia_object / dia_source match radius
INCLUDE_ALL_MATCHES = False    # True = merge dia_sources of every dia_object within the radius
RELIABILITY_MIN = 0.5          # keep dia_sources with reliability >= this
CROSS_WIDTH = 1                # rows/cols averaged for cross sections (odd number)
N_PSF_STARS = 10               # PSF stars shown as stamps
VISIT, DETECTOR = None, None   # set to force the cutout visit; None = highest-SNR detection

butler = Butler(REPO, collections=COLLECTIONS)

point = geom.SpherePoint(ra, dec, geom.degrees)
target = SkyCoord(ra * u.deg, dec * u.deg)
WHERE_POINT = f"POINT({ra}, {dec})"

_registered = {dt.name for dt in butler.registry.queryDatasetTypes()}


def resolve(*candidates):
    """Return the first dataset-type name that exists in this repo (handles old/new names)."""
    for c in candidates:
        if c in _registered:
            return c
    raise LookupError(f"None of {candidates} are registered in {REPO}")


def query_overlapping(dstype):
    """All datasets of `dstype` whose spatial region contains (ra, dec)."""
    dims = butler.get_dataset_type(dstype).dimensions.names
    if {"visit", "detector"} <= set(dims):
        region = "visit_detector_region.region"
    elif "patch" in dims:
        region = "patch.region"
    elif "tract" in dims:
        region = "tract.region"
    elif "visit" in dims:
        region = "visit.region"
    else:
        raise ValueError(f"{dstype} has no spatial dimensions: {dims}")
    return butler.query_datasets(
        dstype,
        where=f"{region} OVERLAPS {WHERE_POINT}",
        with_dimension_records=True,
        limit=None,
        explain=False,
    )


VISIT_IMAGE = resolve("visit_image", "calexp")
DIFF_IMAGE = resolve("difference_image", "goodSeeingDiff_differenceExp")
TEMPLATE_DET = resolve("template_detector", "goodSeeingDiff_templateExp")
DEEP_COADD = resolve("deep_coadd", "deepCoadd_calexp")
TEMPLATE_COADD = resolve("template_coadd", "goodSeeingCoadd")
DIA_OBJECT = resolve("dia_object", "diaObjectTable_tract")
DIA_SOURCE = resolve("dia_source", "diaSourceTable_tract")
try:
    SOURCE = resolve("source", "single_visit_star", "sourceTable_visit", "src")
except LookupError:
    SOURCE = None
print(VISIT_IMAGE, DIFF_IMAGE, TEMPLATE_DET, DEEP_COADD, TEMPLATE_COADD, DIA_OBJECT, DIA_SOURCE)

# %% [markdown]
# ## 1. Visit images (visit + detector) overlapping the position

# %%
visit_refs = query_overlapping(VISIT_IMAGE)
rows = []
for ref in visit_refs:
    rec = ref.dataId.records["visit"]
    rows.append(dict(
        visit=ref.dataId["visit"],
        detector=ref.dataId["detector"],
        band=rec.band if hasattr(rec, "band") else ref.dataId.get("band"),
        physical_filter=rec.physical_filter,
        obs_start=rec.timespan.begin.utc.isot if rec.timespan else None,
        exp_time=rec.exposure_time,
        run=ref.run,
    ))
if not rows:
    raise RuntimeError(f"No {VISIT_IMAGE} overlaps ({ra}, {dec}) in {COLLECTIONS}. "
                       "Check the collection list printed above.")
visits = pd.DataFrame(rows).sort_values("obs_start").reset_index(drop=True)
print(f"{len(visits)} {VISIT_IMAGE} datasets overlap ({ra}, {dec})")
print(visits.groupby("band").size())
visits

# %% [markdown]
# ## 2. Coadds overlapping the position

# %%
coadd_rows = []
for dstype in (DEEP_COADD, TEMPLATE_COADD):
    for ref in query_overlapping(dstype):
        d = ref.dataId
        coadd_rows.append(dict(dataset_type=dstype, skymap=d["skymap"], tract=d["tract"],
                               patch=d["patch"], band=d["band"], run=ref.run))
coadds = pd.DataFrame(coadd_rows).sort_values(["dataset_type", "band", "tract", "patch"])
print(f"{len(coadds)} coadds overlap")
coadds

# %% [markdown]
# ## 3. DIA object and DIA sources
# Prompt Processing / Lasair `diaObjectId`s are not the same as DRP ids, so match on position.

# %%
def load_table(ref):
    t = butler.get(ref)
    df = t.to_pandas() if hasattr(t, "to_pandas") else t
    # DataFrame-stored tables often keep the id (diaObjectId / diaSourceId) as the index.
    if any(n is not None for n in df.index.names):
        df = df.reset_index()
    return df


def id_col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"None of {cands} in columns: {list(df.columns)}")


def radec_cols(df):
    for r, d in (("ra", "dec"), ("coord_ra", "coord_dec")):
        if r in df.columns and d in df.columns:
            return r, d
    raise KeyError("no ra/dec columns found")


def nearby(df, radius_arcsec):
    rcol, dcol = radec_cols(df)
    sep = SkyCoord(df[rcol].values * u.deg, df[dcol].values * u.deg).separation(target).arcsec
    out = df.loc[sep < radius_arcsec].copy()
    out["sep_arcsec"] = sep[sep < radius_arcsec]
    return out.sort_values("sep_arcsec")


dia_obj_all = pd.concat([load_table(r) for r in query_overlapping(DIA_OBJECT)], ignore_index=True)
dia_obj_match = nearby(dia_obj_all, MATCH_RADIUS_ARCSEC)
OBJ_IDCOL = id_col(dia_obj_all, "diaObjectId", "dia_object_id", "id")
print(f"dia_object id column: {OBJ_IDCOL}")
print(f"OBJ_ID {OBJ_ID} present in DRP dia_object table: "
      f"{OBJ_ID in set(dia_obj_all[OBJ_IDCOL])}")
display(dia_obj_match)

dia_src_all = pd.concat([load_table(r) for r in query_overlapping(DIA_SOURCE)], ignore_index=True)
SRC_OBJCOL = id_col(dia_src_all, "diaObjectId", "dia_object_id")
if len(dia_obj_match):
    # Take ids from the column itself: .iloc[0] on the row upcasts int64 ids to float64
    # and corrupts 18-digit ids.
    ids = dia_obj_match[OBJ_IDCOL].astype("int64").tolist()
    use_ids = ids if INCLUDE_ALL_MATCHES else ids[:1]
    dia_src = dia_src_all[dia_src_all[SRC_OBJCOL].astype("int64").isin(use_ids)].copy()
    for i, s in zip(ids, dia_obj_match["sep_arcsec"]):
        sel = dia_src_all[SRC_OBJCOL].astype("int64") == i
        rel_txt = ""
        if "reliability" in dia_src_all:
            r = dia_src_all.loc[sel, "reliability"]
            rel_txt = (f"  reliability median={r.median():.2f} max={r.max():.2f} "
                       f"n>={RELIABILITY_MIN}: {(r >= RELIABILITY_MIN).sum()}")
        print(f"  diaObjectId {i}  sep={s:.2f}\"  n_dia_source={sel.sum()}{rel_txt}"
              f"{'  <- used' if i in use_ids else ''}")
    print(f"-> {len(dia_src)} dia_sources")
else:
    print("No dia_object within radius; falling back to positional dia_source match")
    dia_src = nearby(dia_src_all, MATCH_RADIUS_ARCSEC)

# Reliability cut (ML real/bogus score on each dia_source).
if "reliability" in dia_src:
    n0 = len(dia_src)
    passing = dia_src[dia_src["reliability"] >= RELIABILITY_MIN].copy()
    print(f"Reliability >= {RELIABILITY_MIN}: kept {len(passing)} of {n0} dia_sources")
    if len(passing):
        dia_src = passing
    else:
        print(f"WARNING: no dia_source passes reliability >= {RELIABILITY_MIN} "
              f"(max = {dia_src['reliability'].max():.2f}). Continuing with ALL {n0} detections "
              "so the plots still render -- treat them as low-reliability.")
else:
    print(f"WARNING: no 'reliability' column in {DIA_SOURCE}; no reliability cut applied")

# Flux columns. dia_source has no calibFlux in the standard DPDD schema; if missing we
# fall back to scienceFlux (direct-image PSF flux) and say so.
FLUX_CANDIDATES = {"AP": ["apFlux"], "Calib": ["calibFlux", "scienceFlux"], "PSF": ["psfFlux"]}
FLUX_COLS = {}
for label, cands in FLUX_CANDIDATES.items():
    col = next((c for c in cands if c in dia_src.columns), None)
    if col is None:
        print(f"WARNING: no column for {label} flux (tried {cands})")
        continue
    if col != cands[0]:
        print(f"NOTE: '{cands[0]}' not in {DIA_SOURCE}; using '{col}' for {label}")
    FLUX_COLS[label] = col
print("Flux columns:", FLUX_COLS)

time_col = "midpointMjdTai" if "midpointMjdTai" in dia_src.columns else None
if time_col is None:  # fall back to visit start times
    t0 = {r.dataId["visit"]: r.dataId.records["visit"].timespan.begin.tai.mjd for r in visit_refs}
    dia_src["midpointMjdTai"] = dia_src["visit"].map(t0)
    time_col = "midpointMjdTai"
dia_src = dia_src.sort_values(time_col).reset_index(drop=True)
dia_src[["diaSourceId", "visit", "detector", "band", time_col]
        + (["reliability"] if "reliability" in dia_src else [])
        + [f"{c}{s}" for c in FLUX_COLS.values() for s in ("", "Err") if f"{c}{s}" in dia_src]]

# %% [markdown]
# ## 4. Cutouts (visit / template / difference) + PSF model scaled to each flux

# %%
if VISIT is None:
    snr = (dia_src["psfFlux"] / dia_src["psfFluxErr"]).abs()
    best = dia_src.loc[snr.idxmax()]
    VISIT, DETECTOR = int(best["visit"]), int(best["detector"])
else:
    best = dia_src[(dia_src.visit == VISIT) & (dia_src.detector == DETECTOR)].iloc[0]
data_id = dict(visit=VISIT, detector=DETECTOR)
print(f"Cutouts from visit={VISIT} detector={DETECTOR} band={best.get('band')}")


def cutout(dstype):
    full_bbox = butler.get(f"{dstype}.bbox", data_id)
    wcs = butler.get(f"{dstype}.wcs", data_id)
    xy = wcs.skyToPixel(point)
    half = CUTOUT_PX // 2
    bbox = geom.Box2I(geom.Point2I(int(round(xy.x)) - half, int(round(xy.y)) - half),
                      geom.Extent2I(CUTOUT_PX, CUTOUT_PX))
    bbox.clip(full_bbox)
    return butler.get(dstype, data_id, parameters={"bbox": bbox}), xy


def extent(bbox):
    return (bbox.getMinX() - 0.5, bbox.getMaxX() + 0.5, bbox.getMinY() - 0.5, bbox.getMaxY() + 0.5)


sci, xy = cutout(VISIT_IMAGE)
tmpl, _ = cutout(TEMPLATE_DET)
diff, _ = cutout(DIFF_IMAGE)

psf = butler.get(f"{VISIT_IMAGE}.psf", data_id)
psf_img = psf.computeImage(geom.Point2D(xy))
psf_unit = psf_img.array / psf_img.array.sum()   # normalised to unit flux

fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)

# Row 1: science / template / difference. Science & template share a stretch.
st_norm = ImageNormalize(sci.image.array, interval=ZScaleInterval(), stretch=AsinhStretch(0.1))
diff_norm = ImageNormalize(diff.image.array, interval=ZScaleInterval())
for ax, exp, title, norm in ((axes[0, 0], sci, f"{VISIT_IMAGE}", st_norm),
                             (axes[0, 1], tmpl, f"{TEMPLATE_DET}", st_norm),
                             (axes[0, 2], diff, f"{DIFF_IMAGE}", diff_norm)):
    im = ax.imshow(exp.image.array, origin="lower", cmap="gray", norm=norm, extent=extent(exp.getBBox()))
    ax.plot(xy.x, xy.y, "+", color="tab:red", ms=14, mew=1.5)
    ax.set_title(f"{title}\nvisit {VISIT} det {DETECTOR}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="nJy / pix")

# Row 2: PSF model × each flux, on the difference-image stretch for direct comparison.
for ax, (label, col) in zip(axes[1], FLUX_COLS.items()):
    flux = best[col]
    model = psf_unit * flux
    im = ax.imshow(model, origin="lower", cmap="gray", norm=diff_norm, extent=extent(psf_img.getBBox()))
    ax.set_xlim(axes[0, 2].get_xlim()); ax.set_ylim(axes[0, 2].get_ylim())
    ax.plot(xy.x, xy.y, "+", color="tab:red", ms=14, mew=1.5)
    ax.set_title(f"PSF model × {label} flux\n{col} = {flux:,.0f} nJy")
    fig.colorbar(im, ax=ax, shrink=0.8, label="nJy / pix")
for ax in axes[1, len(FLUX_COLS):]:
    ax.axis("off")

fig.suptitle(f"OBJ_ID {OBJ_ID}  (ra, dec) = ({ra}, {dec})")
plt.show()

# %% [markdown]
# ## 4b. Visit-image PSF model and the stars used to build it
# PSF stars are the `calib_psf_used` sources in the visit's source catalog (`calib_psf_reserved`
# stars were held out of the fit, so they are a fair test of the model).

# %%
def pick(df, *cands):
    return next((c for c in cands if c in df.columns), None)


def psf_model_on(bbox, psf_, x, y, flux):
    """PSF model at (x, y), scaled to `flux`, rendered on the pixel grid of `bbox`."""
    pimg = psf_.computeImage(geom.Point2D(x, y))
    out = np.zeros((bbox.getHeight(), bbox.getWidth()))
    ov = geom.Box2I(pimg.getBBox())
    ov.clip(bbox)
    if ov.isEmpty():
        return out
    out[ov.getMinY() - bbox.getMinY(): ov.getMaxY() - bbox.getMinY() + 1,
        ov.getMinX() - bbox.getMinX(): ov.getMaxX() - bbox.getMinX() + 1] = pimg[ov].array
    return out * flux / pimg.array.sum()


fwhm = psf.computeShape(geom.Point2D(xy)).getDeterminantRadius() * 2.355
kimg = psf.computeKernelImage(geom.Point2D(xy))

full_img = butler.get(f"{VISIT_IMAGE}.image", data_id)
full_bb = full_img.getBBox()

stars = pd.DataFrame()
if SOURCE is not None:
    sdims = butler.get_dataset_type(SOURCE).dimensions.names
    sid = data_id if "detector" in sdims else dict(visit=VISIT)
    src_cat = load_table(butler.find_dataset(SOURCE, sid))
    if "detector" in src_cat and "detector" not in sdims:
        src_cat = src_cat[src_cat["detector"] == DETECTOR]
    XC, YC = pick(src_cat, "x", "slot_Centroid_x", "centroid_x"), pick(src_cat, "y", "slot_Centroid_y", "centroid_y")
    FC = pick(src_cat, "psfFlux", "slot_PsfFlux_instFlux", "base_PsfFlux_instFlux")
    used = pick(src_cat, "calib_psf_used")
    reserved = pick(src_cat, "calib_psf_reserved")
    if used:
        stars = src_cat[src_cat[used].astype(bool)].sort_values(FC, ascending=False)
        resv = src_cat[src_cat[reserved].astype(bool)] if reserved else src_cat.iloc[:0]
        print(f"{SOURCE}: {len(stars)} PSF stars used, {len(resv)} reserved on detector {DETECTOR}")
    else:
        print(f"No calib_psf_used column in {SOURCE}; columns: {list(src_cat.columns)[:40]} ...")
else:
    print("No source catalog dataset type found; skipping PSF stars")

fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1.3]})
im = axes[0].imshow(kimg.array, origin="lower", cmap="viridis",
                    norm=ImageNormalize(kimg.array, stretch=AsinhStretch(0.02)))
fig.colorbar(im, ax=axes[0], shrink=0.8, label="fraction of flux / pix")
axes[0].set_title(f"PSF model (kernel image) at target\nFWHM ≈ {fwhm:.2f} px")

B = 8  # bin factor for the detector overview
a = full_img.array
ny, nx = (a.shape[0] // B) * B, (a.shape[1] // B) * B
binned = np.nanmean(a[:ny, :nx].reshape(ny // B, B, nx // B, B), axis=(1, 3))
axes[1].imshow(binned, origin="lower", cmap="gray",
               norm=ImageNormalize(binned, interval=ZScaleInterval()),
               extent=(full_bb.getMinX(), full_bb.getMinX() + nx, full_bb.getMinY(), full_bb.getMinY() + ny))
if len(stars):
    axes[1].scatter(stars[XC], stars[YC], s=40, facecolors="none", edgecolors="tab:green",
                    label=f"PSF stars used ({len(stars)})")
    if len(resv):
        axes[1].scatter(resv[XC], resv[YC], s=40, marker="s", facecolors="none",
                        edgecolors="tab:orange", label=f"reserved ({len(resv)})")
axes[1].plot(xy.x, xy.y, "*", color="tab:red", ms=16, mec="k", label="target")
axes[1].set_title(f"{VISIT_IMAGE} visit {VISIT} det {DETECTOR} (binned {B}×)")
axes[1].legend(loc="upper right", fontsize=8)
plt.show()

# Stamps: star / PSF model scaled to the star's psfFlux / residual
if len(stars):
    show = stars.head(N_PSF_STARS)
    fig, axes = plt.subplots(3, len(show), figsize=(1.9 * len(show), 6.2), constrained_layout=True,
                             squeeze=False)
    for j, (_, st) in enumerate(show.iterrows()):
        bb = geom.Box2I(psf.computeImage(geom.Point2D(st[XC], st[YC])).getBBox())
        bb.clip(full_bb)
        data = full_img[bb].array
        model = psf_model_on(bb, psf, st[XC], st[YC], st[FC])
        n = ImageNormalize(data, stretch=AsinhStretch(0.05))
        resid = data - model
        lim = np.nanpercentile(np.abs(resid), 99)
        for i, (arr, kw) in enumerate(((data, dict(norm=n, cmap="gray")),
                                       (model, dict(norm=n, cmap="gray")),
                                       (resid, dict(vmin=-lim, vmax=lim, cmap="RdBu_r")))):
            axes[i, j].imshow(arr, origin="lower", **kw)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[0, j].set_title(f"({st[XC]:.0f},{st[YC]:.0f})\n{st[FC]:,.0f} nJy", fontsize=7)
    for i, lab in enumerate(("star", "PSF model", "star − model")):
        axes[i, 0].set_ylabel(lab)
    fig.suptitle(f"Brightest {len(show)} PSF stars (calib_psf_used)")
    plt.show()

# %% [markdown]
# ## 4c. Cross sections through the target: visit image, template, difference
# Note: the difference image is science − (PSF-matched template), so it is not exactly
# visit − template as plotted here.

# %%
def cuts(arr, bbox, x, y, w=CROSS_WIDTH):
    ix, iy, h = int(round(x)) - bbox.getMinX(), int(round(y)) - bbox.getMinY(), w // 2
    row = np.nanmean(arr[max(iy - h, 0): iy + h + 1, :], axis=0)
    col = np.nanmean(arr[:, max(ix - h, 0): ix + h + 1], axis=1)
    xs = np.arange(bbox.getMinX(), bbox.getMaxX() + 1) - x
    ys = np.arange(bbox.getMinY(), bbox.getMaxY() + 1) - y
    return xs, row, ys, col


from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline

SMOOTH_SIGMA = 1.0   # px; Gaussian smoothing applied identically to data, model and residual


def smooth_line(xs, ys, oversample=10):
    ok = np.isfinite(ys)
    xf = np.linspace(xs[ok].min(), xs[ok].max(), ok.sum() * oversample)
    return xf, make_interp_spline(xs[ok], ys[ok], k=3)(xf)


fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True, sharey=True)
for exp, lab, c in ((sci, "visit image", "tab:blue"), (tmpl, "template", "tab:orange"),
                    (diff, "difference (residual)", "tab:green")):
    arr, bb = exp.image.array, exp.getBBox()
    xs, row, ys, col = cuts(gaussian_filter(np.nan_to_num(arr), SMOOTH_SIGMA), bb, xy.x, xy.y)
    ax_x.plot(*smooth_line(xs, row), color=c, lw=2, label=f"{lab} (smoothed)")
    ax_y.plot(*smooth_line(ys, col), color=c, lw=2, label=f"{lab} (smoothed)")
    xr, rr, yr, cr = cuts(arr, bb, xy.x, xy.y)          # raw pixels as faint points
    ax_x.plot(xr, rr, "o", ms=3, color=c, alpha=0.3)
    ax_y.plot(yr, cr, "o", ms=3, color=c, alpha=0.3)
for ax, d in ((ax_x, "x"), (ax_y, "y")):
    ax.axvline(0, color="tab:red", lw=0.8, ls=":")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(f"Δ{d} from target [px]")
ax_x.set_ylabel(f"nJy / pix (mean of {CROSS_WIDTH} px)")
ax_x.set_title(f"Horizontal cut (Gaussian σ={SMOOTH_SIGMA} px)")
ax_y.set_title(f"Vertical cut (Gaussian σ={SMOOTH_SIGMA} px)")
ax_x.legend(fontsize=8)
fig.suptitle(f"visit {VISIT} det {DETECTOR} band {best.get('band')}")
plt.show()

# %% [markdown]
# ## 4d. Difference image vs fitted PSF model
# Model = difference-image PSF at the dia_source centroid × dia_source psfFlux.

# %%
diff_psf = butler.get(f"{DIFF_IMAGE}.psf", data_id)
XS, YS = pick(dia_src, "x", "centroid_x"), pick(dia_src, "y", "centroid_y")
fx, fy = (best[XS], best[YS]) if XS and YS and np.isfinite(best[XS]) else (xy.x, xy.y)
dbb = diff.getBBox()
d_arr = diff.image.array
d_model = psf_model_on(dbb, diff_psf, fx, fy, best["psfFlux"])
d_resid = d_arr - d_model
print(f"Fit position ({fx:.2f}, {fy:.2f}); psfFlux = {best['psfFlux']:,.0f} ± {best['psfFluxErr']:,.0f} nJy"
      + (f"; reliability = {best['reliability']:.2f}" if "reliability" in best else ""))

fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
for ax, arr, title in zip(axes[0], (d_arr, d_model, d_resid),
                          ("difference image", "fitted PSF model", "difference − model")):
    im = ax.imshow(arr, origin="lower", cmap="gray", norm=diff_norm, extent=extent(dbb))
    ax.plot(fx, fy, "+", color="tab:red", ms=14, mew=1.5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="nJy / pix")

smoothed = {name: gaussian_filter(np.nan_to_num(arr), SMOOTH_SIGMA)
            for name, arr in (("difference", d_arr), ("PSF model", d_model), ("residual", d_resid))}
raw = {"difference": d_arr, "PSF model": d_model, "residual": d_resid}
colors = {"difference": "tab:green", "PSF model": "k", "residual": "tab:purple"}

for ax, k, d in ((axes[1, 0], 0, "x"), (axes[1, 1], 1, "y")):
    for name in ("difference", "PSF model", "residual"):
        xs, row, ys, col = cuts(smoothed[name], dbb, fx, fy)
        xc, yc = (xs, row) if k == 0 else (ys, col)
        ax.plot(*smooth_line(xc, yc), color=colors[name], lw=2, label=f"{name} (smoothed)")
        if name != "PSF model":   # raw pixel values as faint points for reference
            xr, rr, yr, cr = cuts(raw[name], dbb, fx, fy)
            ax.plot(*((xr, rr) if k == 0 else (yr, cr)), "o", ms=3, color=colors[name], alpha=0.3)
    ax.axvline(0, color="tab:red", lw=0.8, ls=":")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(f"Δ{d} from fit position [px]")
    ax.set_ylabel("nJy / pix")
    ax.set_title(f"{'Horizontal' if k == 0 else 'Vertical'} cut (Gaussian σ={SMOOTH_SIGMA} px)")
axes[1, 0].legend(fontsize=8)

yy, xx = np.mgrid[dbb.getMinY():dbb.getMaxY() + 1, dbb.getMinX():dbb.getMaxX() + 1]
r = np.hypot(xx - fx, yy - fy).ravel()
o = np.argsort(r)
axes[1, 2].plot(r, d_arr.ravel(), ".", ms=3, color="tab:green", alpha=0.5, label="difference")
axes[1, 2].plot(r[o], d_model.ravel()[o], "-", color="k", label="PSF model")
rb = np.arange(0, 4 * fwhm + 0.5, 0.5)
med = [np.nanmedian(d_arr.ravel()[(r >= a) & (r < a + 0.5)]) for a in rb[:-1]]
axes[1, 2].plot(rb[:-1] + 0.25, med, "-", color="tab:green", lw=2, label="difference (binned median)")
axes[1, 2].axhline(0, color="0.5", lw=0.8)
axes[1, 2].set_xlim(0, 4 * fwhm)
axes[1, 2].set_xlabel("radius from fit position [px]")
axes[1, 2].set_title("Radial profile")
axes[1, 2].legend()
fig.suptitle(f"Difference image PSF fit — visit {VISIT} det {DETECTOR}")
plt.show()

# %% [markdown]
# ## 5. Light curve: time vs AP / Calib / PSF flux

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}
FLUX_COLOR = {"AP": "#1f77b4", "Calib": "#ff7f0e", "PSF": "#2ca02c"}
BAND_SYMBOL = {"u": "circle", "g": "square", "r": "diamond", "i": "triangle-up",
               "z": "x", "y": "star"}
# scienceFlux is a *direct-image* (total) flux, ~10x larger than the difference fluxes,
# so it goes on a secondary y axis instead of flattening the AP / PSF points.
SECONDARY = {"scienceFlux"}


def lc_figure(df, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    t = Time(df[time_col].values, format="mjd", scale="tai").utc.datetime
    hover = [f"visit {r.visit} det {r.detector}<br>band {r.band}<br>MJD {r[time_col]:.4f}"
             + (f"<br>reliability {r.reliability:.2f}" if "reliability" in df else "")
             for _, r in df.iterrows()]
    df = df.assign(_t=t, _hover=hover)
    for label, col in FLUX_COLS.items():
        sec = col in SECONDARY
        for band, g in df.groupby("band"):
            err = g[f"{col}Err"] if f"{col}Err" in g else None
            fig.add_trace(go.Scatter(
                x=g["_t"], y=g[col], mode="markers",
                name=f"{label} ({col}) · {band}" + (" [right axis]" if sec else ""),
                legendgroup=label,
                marker=dict(symbol=BAND_SYMBOL.get(band, "circle"), color=FLUX_COLOR[label], size=8,
                            line=dict(width=0.5, color="black")),
                error_y=dict(type="data", array=err, visible=err is not None, thickness=1),
                text=g["_hover"],
                hovertemplate="%{text}<br><b>%{y:,.0f} nJy</b><extra>" + f"{label} ({col})</extra>",
            ), secondary_y=sec)
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_yaxes(title_text="Difference flux (AP, PSF) [nJy]", secondary_y=False)
    fig.update_yaxes(title_text="scienceFlux, direct image [nJy]", secondary_y=True, showgrid=False)
    # Drag a box to zoom (x and y); double-click to reset. The range slider is left off on
    # purpose: plotly locks the y axis while it is shown. Preset date windows instead:
    fig.update_xaxes(title_text="Date (UTC)", rangeslider_visible=False,
                     rangeselector=dict(buttons=[
                         dict(count=1, label="1d", step="day", stepmode="backward"),
                         dict(count=7, label="1w", step="day", stepmode="backward"),
                         dict(count=1, label="1m", step="month", stepmode="backward"),
                         dict(count=6, label="6m", step="month", stepmode="backward"),
                         dict(step="all", label="all")]))
    fig.update_yaxes(fixedrange=False)
    fig.update_layout(title=title, dragmode="zoom", hovermode="closest", height=560,
                      legend=dict(groupclick="toggleitem"), template="plotly_white")
    return fig


bands = ",".join(sorted(dia_src["band"].dropna().unique())) if "band" in dia_src else "?"
fig = lc_figure(dia_src, f"{DIA_SOURCE} light curve — OBJ_ID {OBJ_ID} (bands: {bands})")
fig.show(config=PLOTLY_CONFIG)

# %% [markdown]
# ## 6. Visits / detectors that went into the template at this position
# Reads `coaddInputs` from the template_coadd patch(es) covering the position, in the band
# of the cutout visit, and keeps only input detectors whose valid area contains the target.

# %%
TEMPLATE_BAND = best["band"]          # band of the cutout visit; change to inspect another band
MAX_THUMBS = 24                       # cap on per-input cutouts

tmpl_refs = [r for r in query_overlapping(TEMPLATE_COADD) if r.dataId["band"] == TEMPLATE_BAND]
print(f"{len(tmpl_refs)} {TEMPLATE_COADD} patch(es) in {TEMPLATE_BAND}: "
      f"{[(r.dataId['tract'], r.dataId['patch']) for r in tmpl_refs]}")

in_rows, in_recs = [], {}
for ref in tmpl_refs:
    try:
        inputs = butler.get(ref.makeComponentRef("coaddInputs"))
    except Exception as e:
        print(f"  no coaddInputs for {ref.dataId}: {e}")
        continue
    ccds = inputs.ccds.subsetContaining(point, includeValidPolygon=True)
    for rec in ccds:
        key = (int(rec["visit"]), int(rec["ccd"]))
        if key in in_recs:            # same input seen via an overlapping patch
            continue
        in_recs[key] = rec
        in_rows.append(dict(visit=key[0], detector=key[1],
                            weight=rec["weight"] if "weight" in rec.schema.getNames() else np.nan,
                            tract=ref.dataId["tract"], patch=ref.dataId["patch"]))

if not in_rows:
    raise RuntimeError("No template inputs found. If template_coadd is cell-based it may not carry "
                       "coaddInputs; check butler.get_dataset_type(TEMPLATE_COADD).storageClass.")

tmpl_inputs = pd.DataFrame(in_rows)
vrecs = {r.id: r for r in butler.query_dimension_records(
    "visit", data_id={"instrument": "LSSTCam"},
    where=f"visit IN ({','.join(map(str, tmpl_inputs.visit.unique()))})")}
tmpl_inputs["obs_start"] = tmpl_inputs.visit.map(lambda v: vrecs[v].timespan.begin.utc.isot)
tmpl_inputs["mjd"] = tmpl_inputs.visit.map(lambda v: vrecs[v].timespan.begin.tai.mjd)
tmpl_inputs["seeing"] = tmpl_inputs.visit.map(lambda v: getattr(vrecs[v], "seeing", np.nan))
tmpl_inputs = tmpl_inputs.sort_values("mjd").reset_index(drop=True)
print(f"{len(tmpl_inputs)} input detectors ({tmpl_inputs.visit.nunique()} visits) cover the target; "
      f"MJD {tmpl_inputs.mjd.min():.2f} – {tmpl_inputs.mjd.max():.2f}")
print(f"Cutout visit {VISIT} is {'IN' if VISIT in set(tmpl_inputs.visit) else 'not in'} the template.")
tmpl_inputs

# %%
from matplotlib.path import Path

# Polygon (RA, Dec corners) for every template input detector, ordered by date.
order = tmpl_inputs.drop_duplicates(["visit", "detector"]).reset_index(drop=True)
polys = []
for _, r in order.iterrows():
    rec = in_recs[(int(r.visit), int(r.detector))]
    corners = [rec.getWcs().pixelToSky(geom.Point2D(c)) for c in rec.getBBox().getCorners()]
    polys.append(np.array([[c.getRa().asDegrees(), c.getDec().asDegrees()] for c in corners]))

# Overlap density: number of input detectors covering each point of a RA/Dec grid.
allv = np.vstack(polys)
pad = 0.02
gra = np.linspace(allv[:, 0].min() - pad, allv[:, 0].max() + pad, 400)
gdec = np.linspace(allv[:, 1].min() - pad, allv[:, 1].max() + pad, 400)
GR, GD = np.meshgrid(gra, gdec)
pts = np.column_stack([GR.ravel(), GD.ravel()])
density = sum(Path(pg).contains_points(pts).astype(int) for pg in polys).reshape(GR.shape)

from matplotlib.colors import to_hex

# One rainbow colour per outline, ordered by date.
colors = [to_hex(c) for c in plt.cm.rainbow(np.linspace(0, 1, len(polys)))]


def densify(pg, n=25):
    """Closed polygon with n points per edge, so hover works anywhere along the outline."""
    closed = np.vstack([pg, pg[:1]])
    return np.vstack([np.linspace(a, b, n, endpoint=False) for a, b in zip(closed[:-1], closed[1:])]
                     + [closed[-1:]])


fig = go.Figure()
fig.add_trace(go.Heatmap(
    x=gra, y=gdec, z=np.where(density > 0, density, np.nan), colorscale="Viridis", opacity=0.55,
    zmin=0, zmax=density.max(), name="overlap density",
    colorbar=dict(title=dict(text="N overlapping inputs", side="top"), orientation="h",
                  x=0.35, xanchor="center", y=-0.12, yanchor="top", len=0.5, thickness=12),
    hovertemplate="RA %{x:.4f}  Dec %{y:.4f}<br><b>%{z} overlapping inputs</b><extra></extra>"))

ctr = allv.mean(axis=0)
for k, (pg, c, (_, r)) in enumerate(zip(polys, colors, order.iterrows()), start=1):
    hover = (f"<b>#{k}</b>  visit {int(r.visit)} / det {int(r.detector)}<br>{r.obs_start[:19]} UTC"
             f"<br>weight {r.weight:.3g}" + (f"<br>seeing {r.seeing:.2f}\"" if np.isfinite(r.seeing) else ""))
    d = densify(pg)
    fig.add_trace(go.Scatter(
        x=d[:, 0], y=d[:, 1], mode="lines", line=dict(color=c, width=2),
        name=f"{k:>2}: {int(r.visit)} / det {int(r.detector)} ({r.obs_start[:10]})",
        legendgroup=str(k), hoverinfo="text", hovertext=hover, hoverlabel=dict(bgcolor=c)))
    far = pg[np.argmax(np.hypot(*(pg - ctr).T))]
    fig.add_trace(go.Scatter(  # number label at the outer corner, toggles with its outline
        x=[far[0]], y=[far[1]], mode="markers+text", text=[str(k)], legendgroup=str(k), showlegend=False,
        marker=dict(size=18, color="white", line=dict(color=c, width=1.5)),
        textfont=dict(color=c, size=11), hoverinfo="text", hovertext=hover))

n_at_target = sum(Path(pg).contains_point((ra, dec)) for pg in polys)
fig.add_trace(go.Scatter(
    x=[ra], y=[dec], mode="markers", name=f"target ({n_at_target} inputs cover it)",
    marker=dict(symbol="star", size=20, color="red", line=dict(color="black", width=1)),
    hovertemplate=f"target<br>RA {ra:.6f}  Dec {dec:.6f}<br>{n_at_target} inputs<extra></extra>"))

fig.update_xaxes(title="RA [deg]", autorange="reversed", scaleanchor="y",
                 scaleratio=np.cos(np.radians(dec)), showgrid=False)
fig.update_yaxes(title="Dec [deg]", showgrid=False)
fig.update_layout(
    title=f"{len(polys)} template input detectors ({TEMPLATE_BAND}) — overlap density",
    template="plotly_white", height=820, width=1150, margin=dict(b=120), dragmode="pan", hovermode="closest",
    legend=dict(title="# : visit / detector (date)<br><i>click to hide, double-click to isolate</i>",
                font=dict(size=10), itemsizing="constant"))
fig.show(config=PLOTLY_CONFIG)

# %%
# Interactive: template input epochs vs AP / Calib / PSF fluxes in the template band.
lc = dia_src[dia_src["band"] == TEMPLATE_BAND] if "band" in dia_src else dia_src
fig = lc_figure(lc, f"Template epochs vs light curve — band {TEMPLATE_BAND}, "
                    f"{len(tmpl_inputs)} input detectors / {tmpl_inputs.visit.nunique()} visits")

t_in = Time(tmpl_inputs.mjd.values, format="mjd", scale="tai").utc.datetime
for tt in sorted(set(t_in)):
    fig.add_vline(x=tt, line_color="rgba(31,119,180,0.35)", line_width=1)
# Hoverable "rug" markers on the zero line, one per template input.
fig.add_trace(go.Scatter(
    x=t_in, y=np.zeros(len(t_in)), mode="markers", name="template input visit",
    marker=dict(symbol="line-ns-open", size=18, color="rgba(31,119,180,0.9)", line=dict(width=2)),
    text=[f"visit {r.visit} det {r.detector}<br>{r.obs_start[:19]}<br>weight {r.weight:.3g}"
          + (f"<br>seeing {r.seeing:.2f}\"" if np.isfinite(r.seeing) else "")
          for _, r in tmpl_inputs.iterrows()],
    hovertemplate="%{text}<extra>template input</extra>"), secondary_y=False)

fig.add_annotation(
    xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top", showarrow=False,
    align="left", bgcolor="rgba(255,255,255,0.8)", bordercolor="gray",
    text=(f"<b>Template ({TEMPLATE_BAND})</b>: {tmpl_inputs.visit.nunique()} visits, "
          f"{tmpl_inputs.obs_start.min()[:10]} → {tmpl_inputs.obs_start.max()[:10]}<br>"
          f"Cutout visit {VISIT} {'IS' if VISIT in set(tmpl_inputs.visit) else 'is not'} a template input"))

# Flag light-curve epochs taken within 1 day of a template input (risk of self-subtraction).
close = [m for m in lc[time_col] if np.min(np.abs(tmpl_inputs.mjd.values - m)) < 1.0]
if close:
    print(f"NOTE: {len(close)} {TEMPLATE_BAND}-band detection(s) are within 1 day of a template "
          "input epoch -- their difference flux is partly self-subtracted.")
fig.show(config=PLOTLY_CONFIG)

# %%
# Thumbnails of each template input visit_image at the target (same stretch for all).
show = tmpl_inputs.drop_duplicates(["visit", "detector"]).head(MAX_THUMBS)
ncol = 6
nrow = int(np.ceil(len(show) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow), constrained_layout=True,
                         squeeze=False)
norm = None
for ax, (_, r) in zip(axes.flat, show.iterrows()):
    did = dict(visit=int(r.visit), detector=int(r.detector))
    try:
        full_bbox = butler.get(f"{VISIT_IMAGE}.bbox", did)
        pxy = butler.get(f"{VISIT_IMAGE}.wcs", did).skyToPixel(point)
        h = CUTOUT_PX // 2
        bb = geom.Box2I(geom.Point2I(int(round(pxy.x)) - h, int(round(pxy.y)) - h),
                        geom.Extent2I(CUTOUT_PX, CUTOUT_PX))
        bb.clip(full_bbox)
        im = butler.get(VISIT_IMAGE, did, parameters={"bbox": bb}).image.array
        norm = norm or ImageNormalize(im, interval=ZScaleInterval(), stretch=AsinhStretch(0.1))
        ax.imshow(im, origin="lower", cmap="gray", norm=norm, extent=extent(bb))
        ax.plot(pxy.x, pxy.y, "+", color="tab:red", ms=10)
    except Exception as e:  # e.g. visit_image not in these collections
        ax.text(0.5, 0.5, type(e).__name__, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{r.visit}/{r.detector}\n{r.obs_start[:10]}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.flat[len(show):]:
    ax.axis("off")
fig.suptitle(f"Template inputs at target ({TEMPLATE_BAND}), first {len(show)} of {len(tmpl_inputs)}")
plt.show()
