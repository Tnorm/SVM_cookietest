import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

cookie = []
for file in ['svm.pkl', 'ntk.pkl']:
    with open(file, 'rb') as f:
        material = pickle.load(f)
    cookie.append(material)
    print(np.mean(material), len(material))

diff = [x-y for x,y in zip(cookie[1], cookie[0])]

plt.plot(range(len(diff)), diff, label=file)
print(stats.wilcoxon(diff))

plt.show()