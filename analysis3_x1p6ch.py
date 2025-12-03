import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_name = "Maternity_leave_days.csv"

df = pd.read_csv(file_name)

df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

COUNTRY_COL = 'country_or_area'
COMP_RATE_COL = 'compensation_rate'
LEAVE_DAYS_COL = 'maternity_leave_length'


df[COMP_RATE_COL] = pd.to_numeric(df[COMP_RATE_COL], errors="coerce")
df[LEAVE_DAYS_COL] = pd.to_numeric(df[LEAVE_DAYS_COL], errors="coerce")

# Rename the leave column for consistent plotting reference
df = df.rename(columns={
    LEAVE_DAYS_COL: 'maternity_leave_days'
})
LEAVE_DAYS_COL_PLT = 'maternity_leave_days'

# ----------------------------------------------------
# Plot 1: Scatter Plot (Maternity leave days vs Compensation rate)
# ----------------------------------------------------
print("Generating Scatter Plot...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=LEAVE_DAYS_COL_PLT, y=COMP_RATE_COL)
plt.title("Maternity Leave Length vs Compensation Rate")
plt.xlabel("Maternity Leave (days)")
plt.ylabel("Compensation Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Maternity Leave Length vs Compensation Rate.png") 

# -----------------------------------------------------------------------------------------
# Plot 2: Bar Plot (Countries vs Maternity leave days, colored by Compensation Rate)
# -----------------------------------------------------------------------------------------
print("Generating Bar Plot with Reduced Labels...")
plt.figure(figsize=(16, 8))

# Sort by leave days for better visual comparison
df_sorted = df.sort_values(by=LEAVE_DAYS_COL_PLT, ascending=False)

ax = plt.gca()
cmap = plt.cm.plasma
max_rate = df_sorted[COMP_RATE_COL].max(skipna=True)

# Create colors for the bars, handling NaN compensation rates with gray (0.7, 0.7, 0.7, 1)
colors = [cmap(rate / max_rate) if not pd.isna(rate) else (0.7, 0.7, 0.7, 1)
          for rate in df_sorted[COMP_RATE_COL]]

# Bar plot
plt.bar(df_sorted[COUNTRY_COL], df_sorted[LEAVE_DAYS_COL_PLT], color=colors)

# --- X-AXIS LABEL-SKIPPING LOGIC (N=2 for every other) ---
N = 5  # Set N to 2, 3, 5, etc., to control label density
labels = []
for i, country in enumerate(df_sorted[COUNTRY_COL]):
    if i % N == 0:
        labels.append(country)
    else:
        labels.append("")  # Skip label
# ---------------------------------------------------------

# Apply the custom labels and rotation
plt.xticks(
    ticks=np.arange(len(df_sorted)), # Create a tick position for every bar
    labels=labels,                   # Apply the generated labels (some blank)
    rotation=90,
    ha='right',
    fontsize=8
)

plt.title("Countries vs Maternity Leave Length (Color = Compensation Rate)", fontsize=14)
plt.ylabel("Maternity Leave (days)")
plt.xlabel("Country or Area")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_rate))
sm.set_array([]) # Required initialization
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
cbar.set_label("Compensation Rate (%)")

plt.tight_layout()
plt.savefig("Countries vs Maternity Leave Length.png") 
plt.show()
