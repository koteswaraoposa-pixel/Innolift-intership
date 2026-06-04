import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Loan_Default.csv")

avg=df.groupby("Region")["income"].mean()

plt.figure(figsize=(8,5))

plt.bar(
    avg.index,
    avg.values,
    color=["red","green","blue","orange"]
)

plt.axhline(
    df["income"].mean(),
    linestyle="--",
    color="black",
    label="Overall Mean"
)

plt.title("Average Income by Region")
plt.xlabel("Region")
plt.ylabel("Income")
plt.legend()

plt.show()
