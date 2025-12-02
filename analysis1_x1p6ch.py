import pandas as pd

# Load the dataset and skip the second row with "%"
df = pd.read_csv("Teaching staff - Data.csv", skiprows=[1])

# Clean numeric columns (convert percentages to floats)
numeric_cols = [
    "Female teachers, Primary education",
    "Female teachers, Secondary education",
    "Female teachers, Tertiary education"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# -------------------------------------------
# Q1: In which education level do women (globally) occupy the highest percentage?
# -------------------------------------------

# Compute the mean percentage for each education level
mean_levels = df[numeric_cols].mean().sort_values(ascending=False)

print("Q1: Global highest female participation by level:")
print(mean_levels)

# -------------------------------------------
# Q2: Which countries have the highest women participation in education (combined)?
# -------------------------------------------

# Compute an average across the three levels for each country
df["Female_avg"] = df[numeric_cols].mean(axis=1)

# Sort countries by highest female participation
top_countries = df[["Country or area", "Female_avg"]].sort_values(
    by="Female_avg", ascending=False
)

print("\nQ2: Countries with the highest average female representation:")
print(top_countries.head(10))

