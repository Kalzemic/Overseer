import pandas as pd
import matplotlib.pyplot as plt

metrics_df_lst = [pd.read_csv(f'results_matmul_{i}.csv') for i in range(1, 2)]
metrics_df = pd.concat(metrics_df_lst, ignore_index=True)
metrics_df["id"] = range(1, len(metrics_df) + 1)

def get_series(col, cond):
    return metrics_df.loc[metrics_df['condition'] == cond, col].dropna()

def remove_outliers_iqr(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return s[(s >= lo) & (s <= hi)]

runtime_local    = remove_outliers_iqr(get_series('elapsed_s', 'overseer_local'))
runtime_baseline = get_series('elapsed_s', 'baseline')
#runtime_remote   = get_series('elapsed_s', 'overseer_remote')


majflt_local    = remove_outliers_iqr(get_series('majflt', 'overseer_local'))
majflt_baseline = get_series('majflt', 'baseline')
#majflt_remote   = get_series('majflt', 'overseer_remote')


plt.hist(runtime_local, label='Overseer Local', alpha=0.6)
plt.hist(runtime_baseline, label='Baseline', alpha=0.6)
#plt.hist(runtime_remote, label='Overseer Remote', alpha=0.6)

plt.legend()
plt.title('Runtime')
plt.xlabel('elapsed_s')
plt.savefig('runtimeMatmul.png')
plt.close()

plt.hist(majflt_local, label='Overseer Local', alpha=0.6)
plt.hist(majflt_baseline, label='Baseline', alpha=0.6)
#plt.hist(majflt_remote, label='Overseer Remote', alpha=0.6)

plt.legend()
plt.title('Major Faults')
plt.xlabel('majflt')
plt.savefig('MajorFaultsMatmul.png')
plt.close()