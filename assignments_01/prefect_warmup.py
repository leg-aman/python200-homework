# Pipeline Question 1
import numpy as np
import pandas as pd
from prefect import task, flow

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
@task
def create_series(arr):
    values = pd.Series(arr)
    return values
@task
def clean_data(series):
    return series.dropna()
@task
def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary
@flow
def pipeline_flow():
    values = create_series(arr)
    values_cleaned = clean_data(values)
    summary = summarize_data(values_cleaned)
    for key, value in summary.items():
        print(f"{key}: {value}")   

if __name__ == "__main__":
    pipeline_flow()

# This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
# This pipeline is very simple, involving only a few small functions and a minimal dataset. It doesn’t require features like scheduling, retries, monitoring,
#  or complex orchestration. Adding Prefect in this case introduces unnecessary complexity without much benefit.

# Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# Prefect becomes valuable in more realistic scenarios, such as: Working with large datasets that take significant time to process
# Handling pipelines that rely on external APIs or databases, where failures can occur
# Running workflows on a schedule (e.g., daily or weekly jobs)
# Managing multi-step pipelines with task dependencies
# Needing robust logging, monitoring, and automatic retries