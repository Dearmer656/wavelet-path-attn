from chart_common import make_chart

lengths = [512, 2048, 4096, 8192, 12288, 16384]
series = [
    ("PaTH-only",  "#000000", "--", 2.0, 6, [0.8158, 0.8119, 0.8047, 0.7859, 0.7544, 0.7136], "s",
     [0.0014, 0.0015, 0.0023, 0.0045, 0.0069, 0.0140]),
    ("K1 ρ=128",         "#EE7733", "-",  2.25, 7, [0.8159, 0.8124, 0.8083, 0.7944, 0.7753, 0.7510], "o",
     [0.0027, 0.0025, 0.0023, 0.0057, 0.0088, 0.0143]),
    ("K1 ρ=256",         "#009988", "-",  2.25, 7, [0.8158, 0.8119, 0.8069, 0.7945, 0.7743, 0.7488], "^",
     [0.0013, 0.0009, 0.0016, 0.0017, 0.0031, 0.0049]),
    ("K3 [128,256,384]", "#33BBEE", "-",  2.9, 9, [0.8158, 0.8109, 0.8066, 0.7908, 0.7678, 0.7372], "*",
     [0.0003, 0.0012, 0.0019, 0.0019, 0.0063, 0.0132]),
]

make_chart(
    title="GPT-2 medium: HotpotQA-Long",
    ylabel="HotpotQA-Long F1",
    x_labels=[f"L{l}" for l in lengths],
    series=series,
    ylim=(0.69, 0.835),
    yticks=[0.70,0.72,0.74,0.76,0.78,0.80,0.82],
    out_path="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/paper_results/scale_sweep_charts/medium_hotpot_sweep.pdf"
)
