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

        df = pd.read_csv(path)

        df["year"] = year   # add year column

        all_data.append(df)

        logger.info(f"Loaded {file}")

    merged = pd.concat(all_data)

    merged.to_csv("assignments_01/outputs/merged_happiness.csv", index=False)

    logger.info("Saved merged file")

    return merged