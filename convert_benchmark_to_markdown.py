import json

import pandas as pd


def convert_benchmark_to_dataframe(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    benchmarks = data["benchmarks"]

    # Create a list of dictionaries to hold the benchmark data
    benchmark_data = []
    for benchmark in benchmarks:

        shape, sigma, seed = benchmark["params"]["data_params"]
        benchmark_data.append(
            {
                "Benchmark name": benchmark["name"][21:-1],
                "data_params": f"shape: {tuple(shape)}, sigma: {sigma}",
                "Min time (s)": benchmark["stats"]["min"],
                "Med time (s)": benchmark["stats"]["median"],
                "Max time (s)": benchmark["stats"]["max"],
                "Mean time (s)": benchmark["stats"]["mean"],
                "StdDev (s)": benchmark["stats"]["stddev"],
            }
        )

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(benchmark_data)
    return df


# Load the benchmark results into a DataFrame
df = convert_benchmark_to_dataframe(
    ".benchmarks/Windows-CPython-3.11-64bit/0006_dfc93d3e0a6673a89f026d1e0f86695f650d8434_20240905_095459_uncommited-changes.json"
)


# Create a separate table for each data param
for data_param in df["data_params"].unique():
    data_param_df = df[df["data_params"] == data_param]
    print(f"### {data_param}")

    data_param_df = data_param_df.drop(columns=["data_params"])
    data_param_df = data_param_df.sort_values("Med time (s)")

    data_param_df["Relative"] = (
        (data_param_df["Med time (s)"] / data_param_df["Med time (s)"].min()).round().astype(int)
    )
    # MOve columns second
    data_param_df = data_param_df[
        [
            "Benchmark name",
            "Med time (s)",
            "Relative",
            "Min time (s)",
            "Max time (s)",
            "Mean time (s)",
        ]
    ]

    print(data_param_df.to_markdown(index=False, floatfmt=".3f"))
