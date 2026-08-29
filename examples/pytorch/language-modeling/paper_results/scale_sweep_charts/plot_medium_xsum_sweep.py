from chart_common import make_chart

lengths = [512, 1024, 1536]
series = [
    ("PaTH-only (s44)",  "#000000", "--", 2.0, 6, [0.4622, 0.4338, 0.3925], "s"),
    ("K1 ρ=128",         "#EE7733", "-",  2.25, 7, [0.4623, 0.4346, 0.3924], "o",
     [0.0003, 0.0013, 0.0012]),
    ("K1 ρ=256",         "#009988", "-",  2.25, 7, [0.4624, 0.4351, 0.3944], "^",
     [0.0005, 0.0011, 0.0008]),
    ("K3 [128,256,384]", "#33BBEE", "-",  2.9, 9, [0.4616, 0.4360, 0.3918], "*",
     [0.0003, 0.0006, 0.0033]),
]

make_chart(
    title="GPT-2 medium: Filtered XSum",
    ylabel="ROUGE-1",
    x_labels=[f"L{l}" for l in lengths],
    series=series,
    ylim=(0.385, 0.47),
    yticks=[0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46],
    out_path="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/paper_results/scale_sweep_charts/medium_xsum_sweep.pdf"
)
