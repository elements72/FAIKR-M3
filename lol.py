# %%
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


# %%
model = BayesianNetwork(
    [
        ("blueKills", "blueGoldDiff"),
        ("blueDeaths", "blueGoldDiff"),
        ("blueGoldDiff", "blueWins"),
        ("blueDragons", "blueWins"),
    ]
)


# %%
import pandas as pd
import numpy as np

data = pd.read_csv("high_diamond_ranked_10min.csv")

# %%
subset = data.loc[:, ["blueKills", "blueWins", "blueDeaths", "blueDragons", "blueGoldDiff"]]

original_subset = subset

print(subset)

# %%
discretized_kills = pd.cut(original_subset["blueKills"], bins=[0, 3, 7, 100], labels=["Low", "Medium", "High"], include_lowest = True)
discretized_deaths = pd.cut(original_subset["blueDeaths"], bins=[0, 3, 7, 100], labels=["Low", "Medium", "High"], include_lowest = True)
discretized_dragons = pd.cut(original_subset["blueDragons"], bins=[-1, 0, 1, 2], labels=["Low", "Medium", "High"], include_lowest = True)
discretized_gold_diff = pd.cut(original_subset["blueGoldDiff"], bins=[-np.inf, -2500, 2500, np.inf], labels=["Negative", "Neutral", "Positive"], include_lowest = True)


subset["blueKills"] = discretized_kills
subset["blueDeaths"] = discretized_deaths
subset["blueDragons"] = discretized_dragons
subset["blueGoldDiff"] = discretized_gold_diff





# %%
print(subset)
print(pd.value_counts(subset["blueKills"]))
print(pd.value_counts(subset["blueDeaths"]))
print(pd.value_counts(subset["blueDragons"]))
print(pd.value_counts(subset["blueGoldDiff"]))



# %%
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from IPython.display import display, HTML

display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))

model.cpds = []

model.fit(data=subset,
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=10,
            complete_samples_only=False,
)

print(f"Check model: {model.check_model()=}")

for cpd in model.get_cpds():
    print(f"CPT of {cpd.variable}")
    print(cpd)


