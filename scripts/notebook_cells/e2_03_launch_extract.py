# Launch the E2-clean extraction sweep. Detached daemon thread, config passed
# as a thread argument, idempotent and disk-backed (P10).
E2_EXTRACT_THREAD = e2_launch_extract(E2_CFG)
E2_EXTRACT_THREAD.name, E2_EXTRACT_THREAD.is_alive()
