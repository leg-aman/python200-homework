from prefect import flow, task, get_run_logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
import os

#  load data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()

    folder = "assignments_01/resources/happiness_project"
    all_data = []

    for year in range(2015, 2025):
        file = f"world_happiness_{year}.csv"
        path = os.path.join(folder, file)

        df = pd.read_csv(path, sep=";", decimal=",", engine="python")

        df.columns = df.columns.str.lower().str.replace(" ", "_")

        df = df.rename(columns={"regional_indicator": "region"})
# add year column
        df["year"] = year   

        all_data.append(df)

        logger.info(f"Loaded {file}")

  
    merged = pd.concat(all_data, ignore_index=True)
    merged = merged.reset_index(drop=True)

    merged.to_csv("assignments_01/outputs/merged_happiness.csv", index=False)

    logger.info("Saved merged file")

    return merged
# stats
@task
def stats_task(df):
    logger = get_run_logger()

    logger.info(f"Mean: {df['happiness_score'].mean()}")
    logger.info(f"Median: {df['happiness_score'].median()}")
    logger.info(f"Std: {df['happiness_score'].std()}")

    by_year = df.groupby("year")["happiness_score"].mean()
    logger.info(f"By year:\n{by_year}")

    by_region = df.groupby("region")["happiness_score"].mean()
    logger.info(f"By region:\n{by_region}")

    return by_region
# graphs
@task
def plots_task(df):
    logger = get_run_logger()

    # Histogram
    plt.figure()
    sns.histplot(df["happiness_score"])
    plt.savefig("assignments_01/outputs/happiness_histogram.png")
    logger.info("Saved histogram")

    # Boxplot
    plt.figure()
    sns.boxplot(x="year", y="happiness_score", data=df)
    plt.savefig("assignments_01/outputs/happiness_by_year.png")
    logger.info("Saved boxplot")

    # Scatter
    plt.figure()
    sns.scatterplot(x="gdp_per_capita", y="happiness_score", data=df)
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
    logger.info("Saved scatter")

    # Heatmap
    plt.figure()
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True)
    plt.savefig("assignments_01/outputs/correlation_heatmap.png")
    logger.info("Saved heatmap")
# test
@task
def test_task(df):
    logger = get_run_logger()

    d2019 = df[df["year"] == 2019]["happiness_score"]
    d2020 = df[df["year"] == 2020]["happiness_score"]

    t, p = ttest_ind(d2019, d2020)

    logger.info(f"T = {t}, P = {p}")

    if p < 0.05:
        logger.info("Happiness changed after pandemic")
    else:
        logger.info("No clear change after pandemic")
# correlation
@task
def corr_task(df):
    logger = get_run_logger()

    cols = df.select_dtypes(include="number").columns

    results = []

    for col in cols:
        if col != "happiness_score":
            r, p = pearsonr(df[col], df["happiness_score"])
            logger.info(f"{col}: r={r}, p={p}")
            results.append((col, r, p))

    # Bonferroni
    n = len(results)
    new_alpha = 0.05 / n

    logger.info(f"New alpha: {new_alpha}")

    return results
@task
def summary_task(df, region_data, corr_results):
    logger = get_run_logger()

    logger.info(f"Countries: {df['country'].nunique()}")
    logger.info(f"Years: {df['year'].nunique()}")

    sorted_regions = region_data.sort_values()

    logger.info(f"Bottom 3:\n{sorted_regions.head(3)}")
    logger.info(f"Top 3:\n{sorted_regions.tail(3)}")

    best = max(corr_results, key=lambda x: abs(x[1]))
    logger.info(f"Strongest correlation: {best}")
@flow
def happiness_pipeline():
    df = load_data()
    regions = stats_task(df)
    plots_task(df)
    test_task(df)
    corr = corr_task(df)
    summary_task(df, regions, corr)


if __name__ == "__main__":
    happiness_pipeline()