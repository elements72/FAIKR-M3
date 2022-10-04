# %%
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# %%
model = BayesianNetwork([
    ("killsDiff", "blueGoldDiff"),
    ("minionsDiff", "blueGoldDiff"),
    ("minionsDiff", "blueExperienceDiff"),
    ("towersDiff", "blueGoldDiff"),
    ("towersDiff", "blueWins"),
    ("blueHeralds", "towersDiff"),
    ("redHeralds", "towersDiff"),
    ("dragonsDiff", "blueWins"),
    ("blueExperienceDiff", "blueWins"),
    ("blueGoldDiff", "blueWins")
])

# %%
import pandas as pd
import numpy as np
def create_dataset(path="high_diamond_ranked_10min.csv"):
    columns = ["blueWins", "blueGoldDiff", "blueExperienceDiff", "blueHeralds", "redHeralds"]
    data = pd.read_csv(path)
    dataset = data.loc[:, columns]
    dataset["killsDiff"] = data.apply(lambda row: row["blueKills"]-row["redKills"], axis=1)
    dataset["minionsDiff"] = data.apply(lambda row: row["blueTotalMinionsKilled"]-row["redTotalMinionsKilled"], axis=1)
    dataset["dragonsDiff"] = data.apply(lambda row: row["blueDragons"]-row["redDragons"], axis=1)
    dataset["towersDiff"] = data.apply(lambda row: row["blueTowersDestroyed"]-row["redTowersDestroyed"], axis=1)
    return dataset

# %%
def discretize(dataset):
    dataset["killsDiff"] = pd.cut(dataset["killsDiff"], bins=[-np.inf, -2, 2, np.inf], labels=["Negative", "Neutral", "Positive"], include_lowest = True)
    dataset["minionsDiff"] = pd.cut(dataset["minionsDiff"], bins=[-np.inf, -15, +15, np.inf], labels=["Negative", "Neutral", "Positive"], include_lowest = True)
    dataset["dragonsDiff"] = pd.cut(dataset["dragonsDiff"], bins=[-np.inf, -1, 0, np.inf], labels=["Negative", "Neutral", "Positive"], include_lowest = True)
    dataset["towersDiff"] = pd.cut(dataset["towersDiff"], bins=[-np.inf, -1, 0, np.inf], labels=["Negative", "Neutral", "Positive"], include_lowest = True)
    dataset["blueGoldDiff"] = pd.cut(dataset["blueGoldDiff"], bins=[-np.inf, -2500, 2500, np.inf], labels=["Negative", "Neutral", "Positive"], include_lowest = True)
    dataset["blueExperienceDiff"] = pd.cut(dataset["blueExperienceDiff"], bins=[-np.inf, -500, 500, np.inf], labels=["Negative", "Neutral", "Positive"], include_lowest = True)

    dataset["blueHeralds"] = pd.cut(dataset["blueHeralds"], bins=[-1,0,np.inf], labels=["Low", "High"], include_lowest = True)
    dataset["redHeralds"] = pd.cut(dataset["redHeralds"], bins=[-1,0,np.inf], labels=["Low", "High"], include_lowest = True)

    return dataset

# %% [markdown]
# 

# %%
dataset = create_dataset()
dataset = discretize(dataset)
dataset.head()

for column in dataset:
    print(pd.value_counts(dataset[column]))



# %%
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from IPython.display import display, HTML

display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))

model.cpds = []

model.fit(data=dataset,
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=10,
            complete_samples_only=False,
)
pd.options.display.max_columns = 2000

print(f"Check model: {model.check_model()=}")

for cpd in model.get_cpds():
    print(f"CPT of {cpd.variable}")
    print(cpd)




