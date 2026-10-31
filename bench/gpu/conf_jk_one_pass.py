# PYSCF_CONFIG_FILE that turns on the opt-in one-pass J+K of the gpu4pyscf perf tree (SCF.jk_one_pass), e.g. to
# run upstream test files on that path. gpu4pyscf.__config__ imports pyscf.__config__, which execs this file.
scf_hf_SCF_jk_one_pass = True
