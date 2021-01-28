import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import numpy as np
import os
import json
import warnings

scen = "Scenariusz5"
scen_dir = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\versions-off-SPP\results\Podejscie2"
jsons_dir = os.path.join(scen_dir, scen)

types = ["raw", "smooth"]
m0 = []
sat = 24

for type_range in types:
    file_name = f"{type_range}_data_with_weight_and_elmask_sat{sat}.json"
    # file_name = f"{type_range}_data_for_sat{sat}.json"
    file_path = os.path.join(jsons_dir, file_name)
    
    with open(file_path) as file:
        data = json.load(file)
        
    m0_array = np.zeros((len(data)))  
    m0_prev = None
    
    for epoch, values in data.items():
        m0_array[int(epoch)] = values["m0"] 
        
        if m0_prev != None and np.abs(values["m0"] - m0_prev) > 1:
            print(int(epoch)+7201, np.abs(values["m0"] - m0_prev))
            
        m0_prev = values["m0"] 
       
    m0.append(m0_array)
    
# Column 0 - raw, Column 1 - smoothed   
m0 = np.vstack(m0).T
epochs = [int(epoch) + 7201 for epoch in data.keys()]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epochs, m0[:, 0], color="C2", label="surowe", linewidth=2)
ax.plot(epochs, m0[:, 1], color="C3", label="wygładzone", linewidth=2)
ax.legend(loc='upper left')
ax.set_xlabel("epoka [UTC]")
ax.set_ylabel(r"$m_0$ [m]")
ax.grid(True)
ax.set_xlim(epochs[0]-100, epochs[-1]+100)
ax.set_xticks([epochs[0] + i*1800 for i in range(5)])
ax.set_xticklabels(['2:00', '2:30', '3:00', '3:30', '4:00'])
fig.set_size_inches(h=3, w=6)



def get_residua_sum(x_train, y_train):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        model = np.poly1d(np.polyfit(x_train, y_train, deg=1))
        
    predict_values = model(x_train)
    residua = y_train - predict_values
    sum_residua = np.dot(residua.T, residua)

    return sum_residua


m0_stats = dict()
train_x = np.array(epochs)
delta_vv_list = []

for i in range(m0.shape[1]):
    if i == 0:
        type_ranges = "surowe"
        
    else:
        type_ranges = "wygładzone"
       
    train_y = m0[: , i]
    delta_vv_list.append(round(get_residua_sum(train_x, train_y), 3))
    
    m0_stats[type_ranges] = {"min":  round(np.min(m0[:, i]), 3),
                            "max":   round(np.max(m0[:, i]), 3),
                            "mean":  round(np.mean(m0[:, i]), 3),
                            "std":   round(np.std(m0[:, i]), 3),
                            }
    
delta_vv = round(delta_vv_list[0] - delta_vv_list[1], 3)
m0_stats["delta [vv]"] = delta_vv
img_name = f"{scen}_m0.pgf"
# plt.savefig(os.path.join(jsons_dir, img_name), bbox_inches='tight')