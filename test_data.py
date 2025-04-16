import pandas as pd
data = [
    {"date": "2023-01-01", "history": 100, "onpromotion": 0, "is_holiday": 1, "transactions": 50},
    {"date": "2023-01-02", "history": 120, "onpromotion": 1, "is_holiday": 0, "transactions": 60},
    {"date": "2023-01-03", "history": 110, "onpromotion": 0, "is_holiday": 0, "transactions": 55}
]
df = pd.DataFrame(data)
df.to_excel("sample_data.xlsx", index=False)