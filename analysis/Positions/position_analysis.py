import sys
sys.path.append(r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\versions-off-SPP")

import SPP_v5 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import os
import json
import warnings


def DOP(A, Approx_rec_coor):
    
    Q = np.linalg.inv(np.dot(A.T, A))
    B, L, _ = SPP_v5.hirvonen(Approx_rec_coor[0], Approx_rec_coor[1], Approx_rec_coor[2])
    B = B * np.pi/180 # rad
    L = L * np.pi/180 # rad
    
    R = np.array([[-np.sin(L), -np.sin(B)*np.cos(L), np.cos(B)*np.cos(L)],
                  [ np.cos(L), -np.sin(B)*np.sin(L), np.cos(B)*np.cos(L)],
                  [ 0        ,  np.cos(B)          , np.sin(L)]])   
    
    Q_xyz = Q[0:3, 0:3]
    Q_ENU = np.dot(np.dot(R.T, Q_xyz), R)
    
    diag = np.diagonal(Q)
    diag_ENU = np.diag(Q_ENU)
    
    GDOP = np.sqrt(np.sum(diag))
    PDOP = np.sqrt(np.sum(diag[:3]))
    TDOP = np.sqrt(diag[-1])
    HDOP = np.sqrt(np.sum(diag_ENU[:2]))
    VDOP = np.sqrt(diag_ENU[-1])
    
    DOP_factors = np.array([GDOP, PDOP, TDOP, HDOP, VDOP])
    
    return DOP_factors


def ECEF_to_ENU(receiver_position, X_ref, Y_ref, Z_ref):
    B, L, _ = SPP_v5.hirvonen(X_ref, Y_ref, Z_ref)
    B = B * np.pi/180 # rad
    L = L * np.pi/180 # rad
    
    R = np.array([[-np.sin(L)          ,  np.cos(L)          , 0],
                  [-np.sin(B)*np.cos(L), -np.sin(B)*np.sin(L), np.cos(B)],
                  [ np.cos(B)*np.cos(L),  np.cos(B)*np.sin(L), np.sin(B)]]) 
    
    delta_pos = np.array([receiver_position[0] - X_ref,
                          receiver_position[1] - Y_ref,
                          receiver_position[2] - Z_ref])
    
    enu_pos = np.dot(R, delta_pos)
    
    return enu_pos # ENU


def visualize_position_residua(residua, img_dir, scen, save=False):
    
    epochs_axis = [epoch + 7201 for epoch in range(residua.shape[0])]
    max_val = np.amax(residua)
    min_val = np.amin(residua)
    
    pairs = {"0": {"idxs": [0,0],
                   "color": "C2",
                   "ylabel": "residua E [m]"
                   },
             "1": {"idxs": [1,0],
                   "color": "C2",
                   "ylabel": "residua N [m]"
                   },
             "2": {"idxs": [2,0],
                   "color": "C2",
                   "ylabel": "residua U [m]"
                   }
             }
    
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(h=5.5, w=7)
    
    for i in range(2):
        for key, value in pairs.items():
            x, y = value["idxs"][0], value["idxs"][1]
            
            if i == 1:
                color = "C3"
                channel = 1
                y += 1
            else:
                color = value["color"]
                channel = 0
                
            axs[x, y].plot(epochs_axis, residua[:, int(key), channel], color=color)
            
            if x == 2 and (y == 1 or y == 0):
                if y == 1:  
                    axs[x, y].set(xlabel='epoka [UTC]')
                else:
                    axs[x, y].set(xlabel='epoka [UTC]', ylabel=value["ylabel"])
                    
            else:
                if y == 1:
                    pass
                else:
                    axs[x, y].set(ylabel=value["ylabel"])
                
            axs[x, y].grid(True)
            axs[x, y].set_ylim([1.1*min_val, max_val*1.1])
            axs[x, y].set_xlim(epochs_axis[0]-100, epochs_axis[-1]+100)
            axs[x, y].set_xticks([epochs_axis[0] + i*1800 for i in range(5)])
            axs[x, y].set_xticklabels(['2:00', '2:30', '3:00', '3:30', '4:00'])
            
    
    # plt.rcParams['xtick.labelsize'] = 17
    # plt.rcParams['ytick.labelsize'] = 17
    # plt.rcParams['axes.labelsize'] = 17
    # plt.rcParams["figure.figsize"] = [20,10]

    
    if save:
        img_name = f"{scen}_residua.pgf"
        plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight')
    
    
def visualize_en(enu, img_dir, scen, save=False):
    
    s = 2.2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=enu[:, 0, 0], y=enu[:, 1, 0], color="C2", label="surowe", s=s)
    ax.scatter(x=enu[:, 0, 1], y=enu[:, 1, 1], color="C3", label="wygładzone", s=s)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.set_size_inches(h=4.5, w=4.5)

    
    if save:
        img_name = f"{scen}_EN.pgf"
        plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight')


def visualize_u(enu, img_dir, scen, save=False):
    
    epochs_axis = [epoch + 7201 for epoch in range(enu_array.shape[0])]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(epochs_axis, enu_array[:, 2, 0], color="C2", label="surowe")
    ax.plot(epochs_axis, enu_array[:, 2, 1], color="C3", label="wygładzone")
    ax.set_xlabel('epoka [UTC]')
    ax.set_ylabel('U [m]')
    
    ax.legend(loc="upper left")
    ax.grid(True)
    
    ax.set_xlim(epochs_axis[0]-100, epochs_axis[-1]+100)
    ax.set_xticks([epochs_axis[0] + i*1800 for i in range(5)])
    ax.set_xticklabels(['2:00', '2:30', '3:00', '3:30', '4:00'])
    fig.set_size_inches(h=3, w=6)
    
    if save:
        img_name = f"{scen}_U.pgf"
        plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight')
        
        
def visualize_DOP(dop_values, img_dir, scen, save=False):
    
    epochs_axis = [epoch + 7201 for epoch in range(enu_array.shape[0])]
    label_dict = {0: "GDOP", 1: "PDOP", 2: "TDOP", 3: "HDOP", 4: "VDOP"}
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for dop_value in range(dop_values.shape[1]):  
        ax.plot(epochs_axis, dop_values[:, dop_value], label=label_dict[dop_value])
        ax.set_xlim(epochs_axis[0]-100, epochs_axis[-1]+100)
        ax.set_xticks([epochs_axis[0] + i*1800 for i in range(5)])
        ax.set_xticklabels(['2:00', '2:30', '3:00', '3:30', '4:00'])
    
    ax.set_xlabel('epoka [UTC]')
    ax.set_ylabel('wartości DOP')
    ax.grid(True)
    
    leg = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                         ncol=5, mode="expand", borderaxespad=0.)
        
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    for item in leg.legendHandles:
        item.set_visible(True)

    fig.set_size_inches(h=3, w=6)
    
    if save:
        img_name = f"{scen}_DOP.pgf"
        plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight')
    
    

# -------------------------------------------
scen = "Scenariusz5"
scen_dir = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\versions-off-SPP\results\Podejscie2"
jsons_dir = os.path.join(scen_dir, scen)
output_img_dir = jsons_dir

types = ["raw", "smooth"]

X_ref = 3655333.908
Y_ref = 1403901.088
Z_ref = 5018038.137 

res_xyz_array = []
enu_array     = []
sat = 24

for type_range in types:
    # file_name = f"{type_range}_data_for_sat{sat}.json"
    file_name = f"{type_range}_data_with_weight_and_elmask_sat{sat}.json"
    file_path = os.path.join(jsons_dir, file_name)
    
    with open(file_path) as file:
        data = json.load(file)

    dop_arr     = np.zeros((len(data), 5))
    res_xyz_arr = np.zeros((len(data), 3))
    enu_arr     = np.zeros((len(data), 3))
    
    for epoch, values in data.items():
        rec_pos = np.array(values["Rec_coors"])
        dop = DOP(np.array(values["DesignMatrix"]), rec_pos)
        dop_arr[int(epoch), :] = dop
        res_xyz_arr[int(epoch), :] = np.array([[rec_pos[0] - X_ref], [rec_pos[1] - Y_ref], [rec_pos[2] - Z_ref]]).T
        enu_arr[int(epoch), :] = ECEF_to_ENU(rec_pos, X_ref, Y_ref, Z_ref)

    res_xyz_array.append(res_xyz_arr)
    enu_array.append(enu_arr)
    
        
res_xyz_array = np.dstack(res_xyz_array)
enu_array     = np.dstack(enu_array)
    
visualize_position_residua(enu_array, output_img_dir, scen, save=True)
visualize_en(enu_array, output_img_dir, scen, save=True)
# visualize_u(enu_array, output_img_dir, scen, save=False) 
# visualize_DOP(dop_arr, output_img_dir, scen, save=False)


def get_residua_sum(x_train, y_train):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        model = np.poly1d(np.polyfit(x_train, y_train, deg=1))
        
    predict_values = model(x_train)
    residua = y_train - predict_values
    sum_residua = np.dot(residua.T, residua)

    return sum_residua


def calc_rms(residua):
    return np.sqrt(np.mean(residua**2))

# Calculate basic statistics
train_x = np.array([epoch + 7201 for epoch in range(res_xyz_array.shape[0])])
values_xyz = ["resX", "resY", "resZ"]
values_enu = ["resE", "resN", "resU"]

stats_xyz_dict = dict()
stats_enu_dict = dict()

for i in range(enu_array.shape[-1]):
    if i == 0:
        type_ranges = "surowe"
        
    else:
        type_ranges = "wygładzone"
        
    res_dict_xyz = dict()   
    res_dict_enu = dict()  
    
    for ((idx_xyz, val_xyz), (idx_enu, val_enu)) in zip (enumerate(values_xyz), enumerate(values_enu)):
        xyz_values = res_xyz_array[:, idx_xyz, i]
        enu_values = enu_array[:, idx_enu, i]
        
        res_dict_xyz[val_xyz] = {"min":  round(np.min(xyz_values), 3),
                                 "max":  round(np.max(xyz_values), 3),
                                 "mean": round(np.mean(xyz_values), 3),
                                 "std":  round(np.std(xyz_values), 3)
                                  }
        
        res_dict_enu[val_enu] = {"min":  round(np.min(enu_values), 3),
                                 "max":  round(np.max(enu_values), 3),
                                 "mean": round(np.mean(enu_values), 3),
                                 "std":  round(np.std(enu_values), 3),
                                 "RMS":  round(calc_rms(enu_values), 3)}
        
        
        
    stats_xyz_dict[type_ranges] = res_dict_xyz
    stats_enu_dict[type_ranges] = res_dict_enu


stats = {"XYZ": stats_xyz_dict,
         "ENU": stats_enu_dict}

        
        
