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


scen = "Scenariusz6"
scen_dir = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\versions-off-SPP\results"
jsons_dir = os.path.join(scen_dir, scen)

types = ["raw", "smooth"]
v_hat = []
sat = 24

for type_range in types:
    # file_name =  f"{type_range}_data_for_sat{sat}.json"
    file_name = f"{type_range}_data_with_weight_and_elmask_sat{sat}.json"
    file_path = os.path.join(jsons_dir, file_name)
    
    with open(file_path) as file:
        data = json.load(file)
        
    v_hat_array = np.zeros((len(data)))  
    el = []
    v_hat_prev = None
    
    for epoch, values in data.items():
        v_hat_array[int(epoch)] = values[f"v_hat-sat{sat}"] 
        
        if v_hat_prev != None and np.abs(values[f"v_hat-sat{sat}"] - v_hat_prev) > 0.2:
            print(int(epoch)+7201, np.abs(values[f"v_hat-sat{sat}"] - v_hat_prev))
            
        v_hat_prev = values[f"v_hat-sat{sat}"] 
        el.append(values[f"el-sat{sat}"])
        
    v_hat.append(v_hat_array)
    
# Column 0 - raw, Column 1 - smoothed   
el = np.array(el)
v_hat = np.vstack(v_hat).T
epochs = [int(epoch) + 7201 for epoch in data.keys()]


fig = plt.figure()
# SUBPLOT left y-axis
ax1 = fig.add_subplot(111)
ax1.plot(epochs,  v_hat[:, 0], color="C2", label="surowe")
ax1.plot(epochs,  v_hat[:, 1], color="C3", label="wygładzone")

ax1.set_ylabel(r'$\hat v$ [m]') 
ax1.set_xlabel('epoka [UTC]') 
ax1.xaxis.labelpad = 10
ax1.grid(True)
ax1.legend(loc='upper left', numpoints=1, fancybox=True)

ax1.set_xlim(epochs[0]-100, epochs[-1]+100)
ax1.set_xticks([epochs[0] + i*1800 for i in range(5)])
ax1.set_xticklabels(['2:00', '2:30', '3:00', '3:30', '4:00'])

# SUBPLOT right y-axis
ax1a = fig.add_subplot(111, sharex=ax1, frameon=False) 
ax1a.scatter(epochs, el, color = 'k', marker = '.', label= 'kat elewacji')  
 
# Y-axis 
ax1a.set_ylim(0, 90)
ax1a.set_yticks(range(0, 90+1, 15))
ax1a.yaxis.tick_right()
ax1a.yaxis.set_label_position("right")
ax1a.set_ylabel('kat elewacji [stopnie]')
ax1a.legend(loc='upper right', numpoints=1, fancybox=True)
fig.set_size_inches(h=3, w=6)


def get_residua_sum(x_train, y_train):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        model = np.poly1d(np.polyfit(x_train, y_train, deg=1))
        
    predict_values = model(x_train)
    residua = y_train - predict_values
    sum_residua = np.dot(residua.T, residua)

    return sum_residua


v_hat_stats = dict()
train_x = np.array(epochs)
delta_vv_list = []

for i in range(v_hat.shape[1]):
    if i == 0:
        type_ranges = "surowe"
        
    else:
        type_ranges = "wygładzone"
       
    train_y = v_hat[: , i]
    delta_vv_list.append(round(get_residua_sum(train_x, train_y), 3))
    
    v_hat_stats[type_ranges] = {"min":  round(np.min(v_hat[:, i]), 3),
                                "max":  round(np.max(v_hat[:, i]), 3),
                                "mean": round(np.mean(v_hat[:, i]), 3),
                                "std":  round(np.std(v_hat[:, i]), 3),
                                }
    
    

delta_vv = round(delta_vv_list[0] - delta_vv_list[1], 3)
v_hat_stats["delta [vv]"] = delta_vv

img_name = f"{scen}_vhat_{sat}.pgf"
# plt.savefig(os.path.join(jsons_dir, img_name), bbox_inches='tight')
