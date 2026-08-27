GA_FIG_DIR = pl.ensure_dir(os.path.join(ga.ART, "figures"))
GA_FIG_MAIN = gaa.make_figure(
    GA_SUMMARY, os.path.join(GA_FIG_DIR, "fig_gateA_dose_matched.png"),
    which="massmean")
GA_FIG_REL = gaa.make_figure_rel(
    GA_SUMMARY, os.path.join(GA_FIG_DIR, "fig_gateA_boundary_relative.png"),
    which="massmean")

GA_FIG_CHECK = {}
for _name, _p in (("main", GA_FIG_MAIN), ("rel", GA_FIG_REL)):
    with open(_p, "rb") as _fh:
        _magic = _fh.read(8)
    GA_FIG_CHECK[_name] = {
        "path": _p,
        "bytes": os.path.getsize(_p),
        "png_magic_ok": _magic == b"\x89PNG\r\n\x1a\n",
    }

mo.vstack([
    mo.md("### Figures written"),
    GA_FIG_CHECK,
    mo.image(GA_FIG_MAIN, width=1100),
    mo.image(GA_FIG_REL, width=700),
])
