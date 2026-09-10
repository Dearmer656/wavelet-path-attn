from chart_common import make_chart

lengths = [512, 1024, 1536]
series = [
    ("PaTH-only",  "#000000", "--", 2.0, 6, [0.4304, 0.3965, 0.3531], "s",
     [0.0005, 0.0020, 0.0020]),
    ("K1 ρ=128",         "#EE7733", "-",  2.25, 7, [0.4304, 0.3964, 0.3517], "o",
     [0.0005, 0.0008, 0.0009]),
    ("K1 ρ=256",         "#009988", "-",  2.25, 7, [0.4297, 0.3971, 0.3528], "^",
     [0.0008, 0.0012, 0.0012]),
    ("K3 [128,256,384]", "#33BBEE", "-",  2.9, 9, [0.4293, 0.3977, 0.3515], "*",
     [0.0004, 0.0003, 0.0023]),
]

make_chart(
    title="GPT-2 medium: Filtered XSum",
    ylabel="ROUGE-L",
    x_labels=[f"L{l}" for l in lengths],
    series=series,
    ylim=(0.34, 0.445),
    yticks=[0.35,0.37,0.39,0.41,0.43],
    out_path="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/paper_results/scale_sweep_charts/medium_xsum_sweep.pdf"
)
