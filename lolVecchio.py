# %%
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


# %%
model = BayesianNetwork(
    [
        ("blueGoldDiff", "blueWins"),
        ("blueDragons", "blueWins"),
        ("blueTowersDestroyed", "blueWins"),
        ("redTowersDestroyed")

        ("blueKills", "blueGoldDiff"),
        ("blueDeaths", "blueGoldDiff"),
        ("redTowersDestroyed", "blueGoldDiff"),
        ("blueTowersDestroyed", "blueGoldDiff"),
        ("blueTotalMinionsKilled", "blueGoldDiff"),
        ("redTotalMinionsKilled", "blueGoldDiff")

    ]
)


# %%
import pandas as pd
import numpy as np

data = pd.read_csv("high_diamond_ranked_10min.csv")

# %%
subset = data.loc[:, ["blueKills", "blueWins", "blueDeaths", "blueDragons", "blueGoldDiff", "redTowersDestroyed", "blueTowersDestroyed", "blueTotalMinionsKilled", "redTotalMinionsKilled"]]

original_subset = subset

print(subset)

# %%
discretized_kills = pd.cut(original_subset["blueKills"], bins=[0, 3, 7, 100], labels=["Low", "Medium", "High"], include_lowest = True)
discretized_deaths = pd.cut(original_subset["blueDeaths"], bins=[0, 3, 7, 100], labels=["Low", "Medium", "High"], include_lowest = True)
discretized_dragons = pd.cut(original_subset["blueDragons"], bins=[-1, 0, 1, 2], labels=["Low", "Medium", "High"], include_lowest = True)
discretized_gold_diff = pd.cut(original_subset["blueGoldDiff"], bins=[-np.inf, -2500, 2500, np.inf], labels=["Negative", "Neutral", "Positive"], include_lowest = True)

discretized_tower_red = pd.cut(original_subset["redTowersDestroyed"], bins=[0, 1, 2, 3], labels=["Low", "Medium", "High"], include_lowest = True)
discretized_tower_blue = pd.cut(original_subset["blueTowersDestroyed"], bins=[0, 1, 2, 3], labels=["Low", "Medium", "High"], include_lowest = True)

discretized_minions_blue = pd.cut(original_subset["blueTotalMinionsKilled"], bins=[0, 150, 250, 450], labels=["Low", "Medium", "High"], include_lowest = True)
discretized_minions_red = pd.cut(original_subset["redTotalMinionsKilled"], bins=[0, 150, 250, 450], labels=["Low", "Medium", "High"], include_lowest = True)


subset["blueKills"] = discretized_kills
subset["blueDeaths"] = discretized_deaths
subset["blueDragons"] = discretized_dragons
subset["blueGoldDiff"] = discretized_gold_diff

subset["blueTowersDestroyed"] = discretized_tower_blue
subset["redTowersDestroyed"] = discretized_tower_red

subset["blueTotalMinionsKilled"] = discretized_minions_blue
subset["redTotalMinionsKilled"] = discretized_minions_red






# %%
print(subset)
print(pd.value_counts(subset["blueKills"]))
print(pd.value_counts(subset["blueDeaths"]))
print(pd.value_counts(subset["blueDragons"]))
print(pd.value_counts(subset["blueGoldDiff"]))

print(pd.value_counts(subset["redTowersDestroyed"]))
print(pd.value_counts(subset["blueTowersDestroyed"]))

print(pd.value_counts(subset["blueTotalMinionsKilled"]))
print(pd.value_counts(subset["redTotalMinionsKilled"]))




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


