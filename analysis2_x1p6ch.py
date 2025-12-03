import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Teaching staff - Data.csv", skiprows=1)  

df.columns = [
    "Country",
    "Female_Primary",
    "Female_Secondary",
    "Female_Tertiary"
]

for col in ["Female_Primary", "Female_Secondary", "Female_Tertiary"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Compute male values (100 - female)
df["Male_Primary"] = 100 - df["Female_Primary"]
df["Male_Secondary"] = 100 - df["Female_Secondary"]
df["Male_Tertiary"] = 100 - df["Female_Tertiary"]


# ===========================================================
# Q1 — PLOT 1  
# MEAN percentage of women vs men across all countries
# ===========================================================

mean_female = [
    df["Female_Primary"].mean(),
    df["Female_Secondary"].mean(),
    df["Female_Tertiary"].mean()
]

mean_male = [100 - x for x in mean_female]

levels = ["Primary", "Secondary", "Tertiary"]

plt.figure(figsize=(9,6))
plt.bar(levels, mean_female, label="Women")
plt.bar(levels, mean_male, bottom=mean_female, label="Men")
plt.title("Mean Teacher Representation by Educational Level (Women vs Men)")
plt.ylabel("Percentage")
plt.legend()
plt.savefig("mean_teacher_representation.png")  # can also use .pdf, .jpg, .svg

# ===========================================================
# Q1 — PLOT 2  
# NUMBER OF COUNTRIES where women dominate vs men dominate
# ===========================================================

dominance_counts = {
    "Primary": {
        "Women": (df["Female_Primary"] > 50).sum(),
        "Men": (df["Female_Primary"] < 50).sum()
    },
    "Secondary": {
        "Women": (df["Female_Secondary"] > 50).sum(),
        "Men": (df["Female_Secondary"] < 50).sum()
    },
    "Tertiary": {
        "Women": (df["Female_Tertiary"] > 50).sum(),
        "Men": (df["Female_Tertiary"] < 50).sum()
    },
}

# Prepare plotting
levels = ["Primary", "Secondary", "Tertiary"]
women_dom = [dominance_counts[l]["Women"] for l in levels]
men_dom = [dominance_counts[l]["Men"] for l in levels]

plt.figure(figsize=(9,6))
plt.bar(levels, women_dom, label="Women-dominated countries")
plt.bar(levels, men_dom, bottom=women_dom, label="Men-dominated countries")
plt.title("Number of Countries by Gender Dominance in Teaching")
plt.ylabel("Number of countries")
plt.legend()
plt.savefig("Countries by Gender Dominace in Teaching.png")

# ===========================================================
# Q2  
# COUNTRY-LEVEL male vs female representation (averaged)
# ===========================================================

df["Female_Avg"] = df[["Female_Primary","Female_Secondary","Female_Tertiary"]].mean(axis=1)
df["Male_Avg"] = 100 - df["Female_Avg"]

df_sorted = df.sort_values("Female_Avg", ascending=False)


# Show only every N-th country name to avoid clutter
N = 10  # change to 2,3,5 etc. depending on density
labels = []
for i, country in enumerate(df_sorted["Country"]):
    if i % N == 0:
        labels.append(country)
    else:
        labels.append("")  # skip label

plt.figure(figsize=(12,10))  # larger width for readability
plt.barh(df_sorted["Country"], df_sorted["Female_Avg"], label="Women")
plt.barh(df_sorted["Country"], df_sorted["Male_Avg"], left=df_sorted["Female_Avg"], label="Men")

# Apply filtered labels and font size
plt.yticks(ticks=range(len(df_sorted)), labels=labels, fontsize=8)

plt.gca().invert_yaxis()  # highest at top
plt.title("Countries Ranked by Female Representation in Education")
plt.xlabel("Percentage")
plt.legend()
plt.savefig("Countries Ranked by Female Representation in Education.png")  # can also use .pdf, .jpg, .svg
plt.show()
