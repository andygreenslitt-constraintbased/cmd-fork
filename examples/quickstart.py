from engine.cmd_fork_clean_v2 import CMDForkWorkflow

# Run 1000 decisions with default config
workflow = CMDForkWorkflow(mode="pure_random", n_runs=1000)
workflow.run()
stats = workflow.summary_stats()

print(f"A (Protect): {stats['A_pct']*100:.1f}%")
print(f"B (Risk):    {stats['B_pct']*100:.1f}%")
print(f"C (Context): {stats['C_pct']*100:.1f}%")
