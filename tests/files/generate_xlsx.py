import pandas as pd
import os

df = pd.DataFrame({
    "id": [1,2,3],
    "value": [10,20,30]
})

os.makedirs("tests/files", exist_ok=True)
df.to_excel("tests/files/test.xlsx", index=False)

print("XLSX created: tests/files/test.xlsx")
