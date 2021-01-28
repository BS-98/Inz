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
from tqdm import tqdm
import time
import warnings
import os


def get_ranges(obs_file, sat):
    with open(obs_file, "r") as f:
        data = f.readlines()
        
    LAMBDA_L1 = 0.190293672798365
    raw_ranges = dict()
    epoch_prev = None
    epoch_satellites = []

    for line in tqdm(data):
        linee = line.split(",") 
        sv = linee[3]
        epoch = int(linee[1]) # SOD
    
        if epoch != epoch_prev and epoch_prev != None:
            if sat not in epoch_satellites:
                raw_ranges[epoch-1] = np.nan
                
            epoch_satellites = []           
                
        if sv == sat:
            if linee[7] == "nan":
                raw_ranges[epoch] = np.nan
            else:
                code_range = float(linee[4]) # code-phase (C/A) observation on L1
                phase_range = round(float(linee[7]) * LAMBDA_L1, 3) # carrier-phase observation on L1
                raw_ranges[epoch] = [code_range, phase_range]
                
        epoch_satellites.append(sv)
        epoch_prev = epoch
    
    return raw_ranges


# smoothing
def smoothing(raw_ranges, window):
    n = 1
    smoothed_code_prev = None
    phase_prev = None
    code_prev = None
    smoothed_ranges = dict()
    
    for epoch, rangee in tqdm(raw_ranges.items()):       
        # if nan
        if type(rangee) == float:
            smoothed_ranges[epoch] = rangee 
            smoothed_code_prev = None
            n = 1
            
        else:
            if n == window + 1:
                n = 1
                
            code, phase = rangee[0], rangee[1]
            
            if n == 1 and smoothed_code_prev == None:
                smoothed_code = code
                
            elif np.abs(code - code_prev)/1000 >= 100:
                smoothed_code = code
                n = 1
                
            else:
                smoothed_code = phase + ((n - 1)/n)*(smoothed_code_prev - phase_prev) + (1/n)*(code - phase) # Hatch Filter
            
            smoothed_code_prev = smoothed_code
            phase_prev = phase
            smoothed_ranges[epoch] = smoothed_code
            n += 1
        
        code_prev = rangee[0] if type(rangee) != float else rangee


    return smoothed_ranges


def get_residua(data, degree):
    train_epochs = np.array(data[0])
    train_ranges = np.array(data[1])
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        model = np.poly1d(np.polyfit(train_epochs, train_ranges, deg=degree))
        
    predict_ranges = model(train_epochs)
    residua = train_ranges - predict_ranges
    
    return residua.flatten().tolist()
    

def calculate_residuals(ranges, range_type, fit_range, degree):
    residua_dict = dict()
    range_prev = None
    list_to_fit = [[], []]  
    t = 0
       
    for epoch, ps_range in tqdm(ranges.items()):
        if range_type == "raw":
            try:
                ps_range = ps_range[0]
                
            except TypeError:
                ps_range = ps_range
        
        elif range_type == "smooth":
            pass
                
        # if nan
        if np.isnan(ps_range):
            residua_dict[epoch] = ps_range        
            t = 0        
            continue
            
        else:
            if range_prev != None and (t == fit_range or np.abs(ps_range - range_prev)/1000 >= 100):
                residua = get_residua(list_to_fit, degree)
                update_list = [(e,r) for e,r in zip(list_to_fit[0], residua)]
                residua_dict.update(update_list)
                
                list_to_fit = [[], []]  
                t = 0
                
            list_to_fit[0].append(epoch)
            list_to_fit[1].append(ps_range)
            t += 1
            
        range_prev = ps_range
        
        
    residua = get_residua(list_to_fit, degree)
    update_list = [(e,r) for e,r in zip(list_to_fit[0], residua)]
    residua_dict.update(update_list)
    
    return residua_dict
   
    
def get_ticks(ranges):
    
    data = [[], []]
    for epoch, ps in ranges.items():
        if type(ps) == float:
            continue
        else:
            data[0].append(epoch)
            data[1].append(ps[0])
            
    xticks = []
    xtickslabels = []

    for i in range(data[0][0], data[0][-1]+1, 7200):
        xticks.append(i)
        xtickslabels.append(str(int((i/7200)*2)))
        
    return data, xticks, xtickslabels
    

def show_ranges(ranges, sat, save):
    
    # Show ranges
    # x = list(ranges.keys())
    # y = [r if type(r) == float else r[0] for r in ranges.values()]
    
    data, xticks, xtickslabels = get_ticks(ranges)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=data[0], y=np.array(data[1]), color="C2", label="surowe", s=1.3)
    ax.grid(True)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabels) 
    ax.set_xlabel('epoka [UTC]')
    ax.set_ylabel('pseudoodległość [m]')
    
    ax.legend(loc='upper left')
    fig.set_size_inches(h=2, w=3.5)
    # plt.tight_layout()
    
    # plt.rcParams['xtick.labelsize'] = 31
    # plt.rcParams['ytick.labelsize'] = 31
    # plt.rcParams['axes.labelsize'] = 31
    # plt.rcParams["figure.figsize"] = [15, 15]
    
    if save:
        img_dir = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\versions-off-SPP\results\ranges"
        # img_name = f"sat{sat}_ranges.png"
        # plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight', dpi=300)
        img_name = f"sat{sat}_ranges.pgf"
        plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight')

        
    
    
def show_residua(raw_residua, smoothed_residua, sat, window, ranges, save=False):
    
    # Show residuals
    y_raw = np.array(list(raw_residua.values()))
    y_smoothed = np.array(list(smoothed_residua.values()))
    
    _, xticks, xtickslabels = get_ticks(ranges)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_raw, color='C2', label="surowe")
    ax.plot(y_smoothed, color='C3', label="wygładzone")
    ax.set_xlabel('epoka [UTC]')
    ax.set_ylabel('residua [m]')
    ax.grid(True)
    
    # ax.legend(loc='upper left', prop={'size': 30})
    ax.legend(loc='upper left')

    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabels)
        
    
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    
    # plt.tight_layout()
    # plt.rcParams['xtick.labelsize'] = 40
    # plt.rcParams['ytick.labelsize'] = 40
    # plt.rcParams['axes.labelsize'] = 40
    # plt.rcParams["figure.figsize"] = [20,15]

    # plt.show()
    
    if save:
        img_dir = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\versions-off-SPP\results\ranges"
        # img_name = f"residua_{sat}_window{window}.png"
        img_name = f"residua_{sat}_window{window}.pgf"
        # plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight')


def stats(res):
    
    values = np.array(list(res.values()))
    values_without_nans = values[np.logical_not(np.isnan(values))]
    stats = {"min": np.min(values_without_nans),
             "max": np.max(values_without_nans),
             "mean": np.mean(values_without_nans),
             "std": np.std(values_without_nans),
             "[vv]": np.dot(values_without_nans.T, values_without_nans)}
   
    return stats
    

def main(observation_file, sat_number, window, fit_range, fit_degree):
    print("Get ranges...")
    raw_ranges = get_ranges(observation_file, sat_number)
    
    print("Smooth ranges...")
    time.sleep(0.3)
    smoothed_ranges = smoothing(raw_ranges, window=window)
    
    print("Calculate residuals...")
    time.sleep(0.3)
    raw_res = calculate_residuals(raw_ranges, range_type="raw", fit_range=fit_range, degree=fit_degree)
    smoothed_res = calculate_residuals(smoothed_ranges, range_type="smooth", fit_range=fit_range, degree=fit_degree)
    
    return raw_res, smoothed_res, raw_ranges, smoothed_ranges




if __name__ == "__main__":
    file = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\satellite_data\sep2_array.asc"
    satellite  = "G09"
    window     = 60
    fit_range  = 80
    fit_degree = 2
    
    raw_res, smoothed_res, raw_ranges, smoothed_ranges = main(file, satellite, window, fit_range, fit_degree)
    # show_residua(raw_res, smoothed_res, satellite, window, raw_ranges, save=False)
    # show_ranges(raw_ranges, satellite, save=False)
    stats_raw = stats(raw_res)
    stats_smoothed = stats(smoothed_res)
    
