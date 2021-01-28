from math import sqrt, sin, atan, atan2, cos, degrees
import numpy as np
from tqdm import tqdm
import time
import json
import os

def prepare_data(file):    
    with open(file, "r") as f:
        obs_data = f.readlines()  
        
    LAMBDA_L1 = 0.190293672798365   
    start_epoch = int(obs_data[0].split(",")[1])
    obs_array = np.empty((86400 - start_epoch,32,2))
    obs_array[:] = np.nan
    sv_epochs = dict()
    epoch_sow_prev = None
    epoch_prev = None
    
    print("Gathering ranges...")
    time.sleep(0.2)
    
    for line in tqdm(obs_data):
        linee = line.split(",") 
        sv = int(linee[3].split("G")[1])
        epoch = int(linee[1]) - start_epoch # SOD
        epoch_sow = int(linee[0]) # SOW
                            
        if "nan" in linee or (sv == 6 and epoch_sow >= 527678) or (sv == 10 and epoch_sow >= 530400) or (sv == 1) or (sv == 17 and epoch_sow <= 528400) or (sv == 20 and 528250 <= epoch_sow <= 528573):
            if epoch_sow_prev != epoch_sow and epoch_sow_prev != None:
                sv_epochs[epoch_prev]["SOW"] = epoch_sow_prev
            
            epoch_sow_prev = epoch_sow
            epoch_prev = epoch
            continue 
        
        if epoch_sow_prev != epoch_sow and epoch_sow_prev != None:
            # print(epoch_sow, epoch_sow_prev, epoch_prev)
            sv_epochs[epoch_prev]["SOW"] = epoch_sow_prev
        
        if np.isnan(float(linee[4])):
            pass
        else:
            try:
                sv_epochs[epoch]["satellites"].append(sv)
            except KeyError:
                sv_epochs[epoch] = {"satellites": [sv]}
        
        obs_array[epoch, sv-1, 0] = float(linee[4])
        obs_array[epoch, sv-1, 1] = np.nan if np.isnan(float(linee[7])) else round(float(linee[7]) * LAMBDA_L1, 3) 
        
        epoch_prev = epoch
        epoch_sow_prev = epoch_sow
        
    sv_epochs[epoch]["SOW"] = epoch_sow
    
    return obs_array, sv_epochs, start_epoch


def smoothing(obs_data, window, start_epoch):

    obs_dataa = obs_data.copy()
    smoothed_array = np.empty((86400 - start_epoch,32))
    smoothed_array[:] = np.nan
    
    print("Smoothing ranges...")
    time.sleep(0.2)
    
    for sat in tqdm(range(obs_dataa.shape[1])):        
        # initial values
        epoch = 0
        smoothed_code_prev = None
        phase_prev = None
        code_prev = None
        n = 1
        
        sat_ranges = obs_dataa[:, sat, :]
                
        for ranges in sat_ranges:
            if np.isnan(ranges).all():
                smoothed_array[epoch, sat] = np.nan 
                smoothed_code_prev = None
                n = 1
            
            # if code obs exists and phase obs not
            elif np.isnan(ranges[0]) == False and np.isnan(ranges[1]):
                smoothed_array[epoch, sat] = ranges[0]
                smoothed_code_prev = None
                n = 1
                
            else:
                if n == window + 1:
                    n = 1
                    
                code, phase = ranges[0], ranges[1]
                
                if n == 1 and smoothed_code_prev == None:
                    smoothed_code = code
                    
                elif np.abs(code - code_prev)/1000 >= 100:
                    smoothed_code = code
                    n = 1
                    
                else:
                    smoothed_code = phase + ((n - 1)/n)*(smoothed_code_prev - phase_prev) + (1/n)*(code - phase) # Hatch Filter
                
                smoothed_code_prev = smoothed_code
                phase_prev = phase
                smoothed_array[epoch, sat] = smoothed_code
                n += 1
            
            code_prev = ranges[0] 
            epoch += 1
    

    return np.dstack((obs_data[:, :, 0], smoothed_array)) # raw and smoothed ranges


def get_frame(t_obs, sv, brdc):
    
    file = open(brdc, 'r')
    lines = file.readlines()
    del lines[:5]
    nr_iteration = int(len(lines)/8)
    i = 0
    # frames = []
    
    for _ in range(nr_iteration):
        
        frame = lines[i:i+8]
    
        if int(frame[0][0:2]) == sv:
            
            year   = int(frame[0][3:5]) + 2000
            month  = int(frame[0][6:8])
            day    = int(frame[0][9:11])
            hour   = int(frame[0][12:14])
            minute = int(frame[0][15:17])
            second = float(frame[0][18:22])
            
            SOW = UTC_2_GPS(year,month,day,hour,minute,second)
            # print(frame)
            # print(SOW)
            time_diff = t_obs - SOW #czas obs - czas z depeszy
            # print(time_diff)
            
            if (time_diff >= 0 and time_diff < 7200):
                # frames.append(frame)
                return frame
                
        i += 8
    
    # return frames[-1]
        
def UTC_2_JD(year,month,day,hour,minute,second):

    A1 = abs(second)/60
    B1 = (abs(minute) + A1)/60
    C1 = abs(hour) + B1
    
    if hour < 0:
        D1 = -C1
    else:
        D1 = C1
        
    dzien2 = int(day + D1/24)
    
    if month < 3 :
        rok2 =  year - 1
        miesiac2 = month + 12
    else:
        rok2 = year
        miesiac2 = month
    
    A = int(rok2/100)
    
    if year > 1582:
        B = 2 - A + int(A/4)
    else:
        B = 0
        
    if rok2 < 0:
        C = int((365.25 * rok2) - 0.75)
    else:
        C = int(365.25 * rok2)
        
    D = int(30.6001 * (miesiac2 + 1))
    
    JD = dzien2 + B + C + D + 1720994.5

    
    return JD

def UTC_2_GPS(year,month,day,hour,minute,second):
    GPS_JD = 2444244.5
    NOW_JD = UTC_2_JD(year,month,day,hour,minute,second)
    diff = NOW_JD - GPS_JD
    GPS_WEEK = int(diff/7)
    DOW = float(diff - GPS_WEEK*7)
        
    SOW = int(DOW * 24 * 3600 + hour*3600 + minute*60 + second)
    
    return SOW 

def calc_sat_pos(t, frame):
    GM84     = 3.986005e14
    OMEGAE84 = 7.2921151467e-5 
     
    Crs     = float(frame[1][22:41].replace("D", "E")) 
    delta_n = float(frame[1][41:60].replace("D", "E"))
    M_0     = float(frame[1][60:79].replace("D", "E"))
    
    Cuc     = float(frame[2][3:22].replace("D", "E"))
    e       = float(frame[2][22:41].replace("D", "E"))
    Cus     = float(frame[2][41:60].replace("D", "E"))
    a_sqrt  = float(frame[2][60:79].replace("D", "E"))
    
    toe     = float(frame[3][3:22].replace("D", "E"))
    Cic     = float(frame[3][22:41].replace("D", "E"))
    Omega_0 = float(frame[3][41:60].replace("D", "E"))
    Cis     = float(frame[3][60:79].replace("D", "E"))
    
    i_0     = float(frame[4][3:22].replace("D", "E"))
    Crc     = float(frame[4][22:41].replace("D", "E"))
    omega   = float(frame[4][41:60].replace("D", "E"))
    Omega   = float(frame[4][60:79].replace("D", "E"))
                
    IDOT    = float(frame[5][3:22].replace("D", "E"))
                
    
    tk = t - toe
    a = a_sqrt**2
    n_0 = sqrt(GM84/a**3)
    n = n_0 + delta_n
    Mk = M_0 + n * tk
    Ek = Mk
    
    while True:
        Ek_0 = Ek
        Ek = Mk + e * sin(Ek_0)
        deltaE = abs(Ek - Ek_0)
        if abs(deltaE) < 1.e-15:
            break
                
#    vk = 2 * atan(sqrt((1 + e)/(1 - e)) * tan(Ek/2))
    vk = atan2(sqrt(1. - e**2)*sin(Ek), cos(Ek) - e)
    u = vk + omega
    delta_uk = Cus * sin(2 * u) + Cuc * cos(2 * u)
    delta_rk = Crs * sin(2 * u) + Crc * cos(2 * u)
    delta_ik = Cis * sin(2 * u) + Cic * cos(2 * u) + IDOT * tk
    uk = u + delta_uk
    rk = a * (1 - e * cos(Ek)) + delta_rk
    ik = i_0 + delta_ik
    Omega_k = Omega_0 + (Omega - OMEGAE84) * tk - OMEGAE84 * toe
    x_prim = rk * cos(uk)
    y_prim = rk * sin(uk)
    X = x_prim * cos(Omega_k) - y_prim * cos(ik) * sin(Omega_k)
    Y = x_prim * sin(Omega_k) + y_prim * cos(ik) * cos(Omega_k)
    Z = y_prim * sin(ik)
    
    return np.array((X, Y, Z))

def sat_clock_error(t_rec, frame):
    year   = int(frame[0][3:5]) + 2000
    month  = int(frame[0][6:8])
    day    = int(frame[0][9:11])
    hour   = int(frame[0][12:14])
    minute = int(frame[0][15:17])
    second = float(frame[0][18:22])
    
    toc = UTC_2_GPS(year,month,day,hour,minute,second)
    # print(day, month, year)
    a0 = float(frame[0][22:41].replace("D", "E")) 
    a1 = float(frame[0][41:60].replace("D", "E")) 
    a2 = float(frame[0][60:79].replace("D", "E"))
    
    dt = (t_rec - toc)
    sat_time_corr = a0 + a1*dt + a2*dt
    tx_corr = t_rec - sat_time_corr 
    dt = tx_corr - toc
    sat_time_corr = a0 + a1*dt + a2*dt
    
    return sat_time_corr


def DOP(A, Approx_rec_coor):
    
    Q = np.linalg.inv(np.dot(A.T, A))
    B, L, _ = hirvonen(Approx_rec_coor[0], Approx_rec_coor[1], Approx_rec_coor[2])
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


def geom_range(sat_coor, rec_coor):
    geom_range = np.sqrt((sat_coor[0] - rec_coor[0])**2 +
                         (sat_coor[1] - rec_coor[1])**2 + 
                         (sat_coor[2] - rec_coor[2])**2)
    
    return geom_range


def rel_range(sat_coor, rec_coor):
    "Relativistic Path Range Correction"
    
    c = 299792458.0
    u = 3986004.418*10**8 # Gravitational constant [m3/s2]
    
    sat_geoc_range = geom_range(sat_coor, np.zeros(3))
    rec_geoc_range = geom_range(rec_coor, np.zeros(3))
    geom_dist = geom_range(sat_coor, rec_coor)
    
    rel_dist_corr = ((2*u)/c**2) * np.log((sat_geoc_range + rec_geoc_range + geom_dist)/(sat_geoc_range + rec_geoc_range - geom_dist))
    
    return rel_dist_corr

                    
def sat_coor_rotated(sat_coor, rec_coor):
    omega = 7292115.0*10**(-11) # rad/s
    c = 299792458.0 # m/s
    geom_dist = geom_range(sat_coor, rec_coor)
    delta_t = geom_dist/c
    alfa = omega*delta_t
    R = np.array([[ cos(alfa), sin(alfa), 0],
                  [-sin(alfa), cos(alfa), 0],
                  [0         , 0        , 1]])
    
    coor_rotated = np.dot(R, sat_coor)

    return coor_rotated


def decimalDeg2dms(decimalDeg):
    d = int(decimalDeg)
    m = int((decimalDeg - d) * 60)
    s = (decimalDeg - d - m/60.0) * 3600
    
    return (d,m,s)


def hirvonen(X, Y, Z, a = 6378137., e2 = 0.00669438002290):
    
    r  = sqrt(X**2 + Y**2)         
    B  = atan(Z / (r * (1 - e2)))    
    B_next = B
    i = 0
    
    while True:
        i +=1
        B_prev = B_next
        N = a/(1-e2*(sin(B_prev))**2)**(0.5)
        H  = (r/cos(B_prev))- N
        B_next = atan(Z/(r *(1 - (e2 * (N/(N + H))))))
        B_next = B_next
        
        if abs(B_next - B_prev) <(0.0000001/206265): 
            break
        
    B = B_prev
    L = atan(Y/X)
    N = a/(1-e2*(sin(B))**2)**(0.5)
    H = (r/cos(B))- N
    
    return degrees(B), degrees(L), H
  

def calc_measurement_matrix(sat_coor, approx_rec_coor, pseudorange):

    measurement_matrix = []
    
    for i in range(len(sat_coor)):
        x = (sat_coor[i,0] - approx_rec_coor[0])/pseudorange[i]
        y = (sat_coor[i,1] - approx_rec_coor[1])/pseudorange[i]
        z = (sat_coor[i,2] - approx_rec_coor[2])/pseudorange[i]
        
        measurement_matrix.append([-x, -y, -z, 1])
    
    measurement_matrix = np.array(measurement_matrix)
    
    return measurement_matrix


def azimuth_and_el(sat_coor, rec_coor):
    """
    INPUT:
        sat_coor - satellite coordinates
        rec_coor - receiver coordinates
    OUTPUT:
        azimuth - azimuth to the satellite [rad]
        el      - elevation angle to the satellite [degrees]
    """

    if rec_coor.all() == 0:
        azimuth, el = 0, 0 #?????
    
    else:
        B, L, _ = hirvonen(rec_coor[0], rec_coor[1], rec_coor[2])
        B *= np.pi/180 # rad
        L *= np.pi/180 # rad
        
        u = np.array([ [cos(B)*cos(L) ,  cos(B)*sin(L), sin(B)] ])
        n = np.array([ [-sin(B)*cos(L), -sin(B)*sin(L), cos(B)] ])
        e = np.array([ [-sin(L)       ,  cos(L)       , 0     ] ])
        
        R = sat_coor - rec_coor
        r = np.expand_dims(R/np.linalg.norm(R), axis=1)
        cosz = np.dot(u, r)
        z = np.arccos(cosz)
        el = 90 - degrees(z)
        azimuth = atan2(np.dot(e, r), np.dot(n, r))
        
        if azimuth < 0:
            azimuth += 2*np.pi
    #    azymut = decimalDeg2dms(degrees(azimuth))
    return azimuth, el
    

def iono(rec_coor, azimuth, el, t_GPS):
    """Model Klobuchar.
    
    INPUT:
        rec_coor: XYZ [m]
        azimuth [rad]
        el [degrees]
        t_GPS [sod]
        brdc - navigation file
    OUTPUT:
        Ionospheric delay [m]
    """
    
    c = 299792458.0
    
    if rec_coor.all() == 0:
        return 0
    
    else:
        B, L, _ = hirvonen(rec_coor[0], rec_coor[1], rec_coor[2])
        
        alfa = [4.6566E-09, 1.4901E-08, -5.9605E-08, -5.9605E-08]
        beta = [7.7824E+04, 4.9152E+04, -6.5536E+04, -3.2768E+05]

        e = el/180 #kat elewacji
        psi = 0.0137/(e + 0.11) - 0.022 #kat geocentryczny
        Fi_I = B/180 + psi * np.cos(azimuth) #szerokosc IPP
        
        if Fi_I > 0.416:
            Fi_I = 0.416
        
        elif Fi_I < -0.416:
            Fi_I = -0.416
            
        Lambd_I = L/180 + ((psi * np.sin(azimuth))/np.cos(Fi_I * np.pi)) #dlugosc IPP
        Fi_m = Fi_I + 0.064 * np.cos((Lambd_I - 1.617) * np.pi) #szerokosc geomagnetyczna IPP
        t = 43200 * Lambd_I + t_GPS #czas lokalny
        
        if t > 86400:
            t -= 86400
            
        elif t < 0:
            t += 86400
            
        #Amplituda opóźnienia jonosferycznego
        A_I = alfa[0] + alfa[1]*Fi_m + alfa[2]*(Fi_m**2) + alfa[3]*(Fi_m**3)
        
        if A_I < 0:
            A_I = 0
            
        #Okres opóźnienia jonosferycznego
        P_I = beta[0] + beta[1]*Fi_m + beta[2]*(Fi_m**2) + beta[3]*(Fi_m**3)
        
        if P_I < 72000:
            P_I = 72000
            
        X = (2*np.pi * (t - 50400))/P_I #Faza opóźnienia jonosferycznego
        F = 1 + 16*(0.53 - e)**3 #funkcja mapująca
        
        if X >= np.pi/2:
            I = c * F * 5 * 10**(-9)
        
        elif X < np.pi/2:
            I = c * F * (5*10**(-9) + A_I*(1 - (X**2)/2 + (X**4)/24))
    
        return I


def tropo(rec_coor, el):
    "Model Hopfield."
    
    if rec_coor.all() == 0:
        return 0
    
    else:
        _, _, H = hirvonen(rec_coor[0], rec_coor[1], rec_coor[2])
        N = 29.29 #odstep elipsoidy od geoidy
        H -= N
        
        # H = 390.92 - N
        # el = 15.32
        
        c1 = 77.64
        c2 = -12.96
        c3 = 3.718 * 10**5
        p0 = 1013.25
        t0 = 291.15
        Rh = 0.5
        
        p  = p0 * (1 - 0.0000226 * H)**(5.225)
        t  = t0 - 0.0065 * H
        RH = Rh * np.exp(-0.0006396 * H)    
        e  = 6.11 * RH * 10**((7.5*(t - 273.15))/(t - 35.85))
        
        Nd0 = c1 * p/t
        Nw0 = c2*e/t + c3*e/t**2
        
        hd = 40136 + 148.72 * (t - 273.15)
        hw = 11000
        md = 1/np.sin(np.sqrt(el**2 + 6.25)*np.pi/180)
        mw = 1/np.sin(np.sqrt(el**2 + 2.25)*np.pi/180)
        
        T = (10**(-6)/5) * (Nd0*hd*md + Nw0*hw*mw)
        
        return T


def get_rec_pos(brdc, ranges, epochs_data, type_ranges, end_epoch, weight_mode, el_tresh, output_dir):
    
    config = {"24": ["raw", "smooth"]}
              
    c = 299792458.0
    results = dict()


    for sv, type_ranges in config.items():
        sv = int(sv)
        
        for type_range in type_ranges:
            if type_range == "raw":
                channel = 0
                
            elif type_range == "smooth":
                channel = 1
    
            for epoch, values in tqdm(epochs_data.items()):
                rec_coor_approx = np.array([3655333.4383, 1403901.4369, 5018038.5146])
                rec_clock_offset_approx = 0
                ps_observed = ranges[epoch, [s-1 for s in values["satellites"]], channel]
                t_rec_reception = values["SOW"]
                
                
                while True:
                    rec_coor = rec_coor_approx.copy()
                    rec_clock_offset = rec_clock_offset_approx
                    ps_computed_list = []
                    sat_coor_list    = []
                    geom_range_list  = []
                    
                    if weight_mode:
                        weight_matrix = []
                        
                    sat_idxs = []
                    sv_idx   = None
                    el_dict  = dict()
                    
                    for sat_idx, sat in enumerate(values["satellites"]):             
                        delta_t = ranges[epoch, sat-1, channel]/c # tau
                        t_sat_emission = t_rec_reception - delta_t
                        frame = get_frame(t_sat_emission, sat, brdc)
                        sat_clock_offset = sat_clock_error(t_sat_emission, frame)  
                        T_sat_emission = t_sat_emission - sat_clock_offset #+ rec_clock_offset
                        
                        frame_emission = get_frame(T_sat_emission, sat, brdc)
                        sat_clock_offset_ost = sat_clock_error(T_sat_emission, frame_emission)
        
                        sat_coor = calc_sat_pos(T_sat_emission, frame_emission)
                        sat_coor_rot = sat_coor_rotated(sat_coor, rec_coor)
                        
                        az, el = azimuth_and_el(sat_coor_rot, rec_coor)
                        
                        if el <= el_tresh:
                            el_dict[str(sat)] = np.nan
                            continue
                        
                        el_dict[str(sat)] = el
                        
                        frame_to_calc_v = get_frame(T_sat_emission+0.1, sat, brdc)
                        sat_coor_to_calc_v = calc_sat_pos(T_sat_emission+0.1, frame_to_calc_v)
                        sat_coor_rot_to_calc_v = sat_coor_rotated(sat_coor_to_calc_v, rec_coor)
                        v = (sat_coor_rot_to_calc_v - sat_coor_rot)/0.1
                        rel_sat_clock_offset = -2 * ((np.dot(sat_coor_rot.T, v))/c**2)
                        
                        I = iono(rec_coor, az, el, t_rec_reception)
                        T = tropo(rec_coor, el)
              
                        rel_range_corr = rel_range(sat_coor_rot, rec_coor)
                        geom_eucl = geom_range(sat_coor_rot, rec_coor)
                        geom_dist = geom_eucl + rel_range_corr
                        
                        ps_computed = geom_dist + c*(rec_clock_offset - (sat_clock_offset_ost + rel_sat_clock_offset)) + I + T
                       
                        ps_computed_list.append(ps_computed)
                        sat_coor_list.append(sat_coor_rot)
                        geom_range_list.append(geom_dist)
                        
                        if weight_mode:
                            weight_matrix.append((np.sin(el*np.pi/180))**2)
                            
                        sat_idxs.append(sat_idx)
                        
                        if sat == sv:
                            sv_idx = sat_idxs.index(sat_idx)
                        
                        
                    A = calc_measurement_matrix(np.array(sat_coor_list), rec_coor, geom_range_list)
                    L = ps_observed[sat_idxs] - np.array(ps_computed_list) # OMC
                    
                    if weight_mode:
                        P = np.diag(weight_matrix)
                    else:
                        P = np.identity(L.shape[0]) 
                        
                    Q = np.linalg.inv(np.linalg.multi_dot([A.T, P, A]))
                    X = np.linalg.multi_dot([Q, A.T, P, L])
                    V_hat = L - np.dot(A, X)
                    m0 = np.nan if L.shape[0] - 4 == 0 else np.sqrt((np.linalg.multi_dot([V_hat.T, P, V_hat]))/(L.shape[0] - 4))
                    
                    rec_coor_approx = rec_coor + X[:-1]
                    rec_clock_offset_approx = rec_clock_offset + X[-1]/c
                    
                    error = rec_coor_approx - rec_coor
                    
                    if (abs(error) <= 0.001).all():
                        v_hat = np.nan if sv not in values["satellites"] or sv_idx == None else float(V_hat[sv_idx])
              
                        results[epoch] = {"Rec_coors": rec_coor_approx.tolist(),
                                          "DesignMatrix": A.tolist(),
                                          "rec_clock_offset": float(rec_clock_offset_approx),
                                          "m0": float(m0),
                                          f"v_hat-sat{sv}": v_hat,
                                          f"el-sat{sv}": float(el_dict[str(sv)]) if sv in values["satellites"] else np.nan}
                        break
        
        
                if epoch == end_epoch:
                    if weight_mode:
                        file_name = f"{type_range}_data_with_weight_and_elmask_sat{sv}.json"
                    else:
                        file_name = f"{type_range}_data_for_sat{sv}.json"
        
                    file_path = os.path.join(output_dir, file_name)
                    
                    with open(file_path, 'w') as file:
                        json.dump(results, file, indent=2)
                        
                    break


def get_ranges(obs_file, smooth_window, type_ranges):
    
    raw_ranges, epochs, start_epoch = prepare_data(obs_file)
    
    if type_ranges == "raw":        
        return raw_ranges, epochs
    
    elif type_ranges == "smooth":
        ranges = smoothing(raw_ranges, window=smooth_window, start_epoch=start_epoch)
        return ranges, epochs

def main(obs_file, nav_file, output_dir, smooth_window=360, end_epoch=7200, weight_mode=False, el_mask=0):

    ranges, epochs = get_ranges(obs_file, smooth_window, type_ranges)
    get_rec_pos(nav_file, ranges, epochs, type_ranges, end_epoch, weight_mode, el_mask, output_dir)



if __name__ == "__main__":
    
    file = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\satellite_data\sep2_array_forpos.asc"
    brdc = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\satellite_data\brdc2150.19n" 
    output_dir = r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\versions-off-SPP\results" 
    
    main(brdc, file, output_dir)


    
    
    
    