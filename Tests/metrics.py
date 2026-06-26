import pandas as pd 
import matplotlib.pyplot as plt

metrics_df = pd.read_csv('pilot.csv')

durations = metrics_df['elapsed_s']


# for row in metrics_df.itertuples():
    
#     durations.append(row.elapsed_s)


plt.plot(durations)
plt.show()

