GA_MANIFEST = {
    "experiment": GA_SUMMARY["experiment"],
    "provenance": GA_SUMMARY["provenance"],
    "fingerprint_check": GA_SUMMARY["fingerprint_check"],
    "checkpoints": {
        _l: {"repo": _m["repo"], "branch": _m["branch"], "commit": _m["commit"]}
        for _l, _m in GA_SUMMARY["models"].items()
    },
    "steer_param": GA_SUMMARY["steer_param"],
    "verdict": GA_SUMMARY["verdict"],
    "figures": GA_FIG_CHECK,
    "artifacts": sorted(os.listdir(ga.ART)),
}
GA_MANIFEST_PATH = pl.write_json(
    os.path.join(ga.ART, "gateA_manifest.json"), GA_MANIFEST)

mo.vstack([mo.md("### Manifest"), GA_MANIFEST, mo.md("`%s`" % GA_MANIFEST_PATH)])
