import gate_a_analysis as gaa

GA_SUMMARY = gaa.summarise(art=ga.ART, n_boot=1000, seed=ga.SEED)
GA_SUMMARY_PATH = pl.write_json(os.path.join(ga.ART, "gateA_summary.json"), GA_SUMMARY)

{
    "models_loaded": sorted(GA_SUMMARY["models"].keys()),
    "fingerprint_check": GA_SUMMARY["fingerprint_check"],
    "cos_massmean_logistic": GA_SUMMARY["directions"].get("cos_massmean_logistic"),
    "smoke_all_pass": GA_SUMMARY["smoke"].get("all_pass"),
    "summary_path": GA_SUMMARY_PATH,
}
