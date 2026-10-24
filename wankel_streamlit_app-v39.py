# -*- coding: utf-8 -*-
"""
==============================================================================
WANKEL ENGINE 0D THERMODYNAMIC SIMULATOR - NATURALLY ASPIRATED EDITION (V39)
==============================================================================
شبیه‌ساز ترمودینامیکی و عملکردی پیشرانه ونکل تنفس طبیعی (مزدا RX-8 / Renesis 13B)
نسخه نهایی کالیبره‌شده با نمودار تجربی MAP، تلفات پمپینگ (PMEP)، مصرف سوخت و EGT
==============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Wankel Engine 0D Simulator (NA Edition - V39)",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM STYLING -----------------
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stButton>button { background-color: #D32F2F; color: white; border-radius: 6px; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #B71C1C; color: white; }
    .metric-card { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 10px; }
    .metric-value { font-size: 20px; font-weight: bold; color: #58A6FF; }
    .metric-label { font-size: 12px; color: #8B949E; }
</style>
""", unsafe_allow_html=True)

# ----------------- EXPERIMENTAL THROTTLE LUT -----------------
# Digitized directly from photo_2026-09-12_02-12-11.jpg
EXPERIMENTAL_THROTTLE_LUT = {
    0.0: 0.200, 10.0: 0.503, 20.0: 0.671, 30.0: 0.799, 40.0: 0.878,
    50.0: 0.938, 60.0: 0.967, 70.0: 0.982, 80.0: 0.987, 90.0: 0.992, 100.0: 1.000
}

def get_ratio_local(th_val, lut_dict):
    if not lut_dict:
        return 0.20 + 0.80 * ((th_val / 100.0) ** 1.5)
    sorted_k = sorted(lut_dict.keys())
    if th_val <= sorted_k[0]: return lut_dict[sorted_k[0]]
    if th_val >= sorted_k[-1]: return lut_dict[sorted_k[-1]]
    for i in range(len(sorted_k) - 1):
        if sorted_k[i] <= th_val <= sorted_k[i+1]:
            r1, r2 = lut_dict[sorted_k[i]], lut_dict[sorted_k[i+1]]
            return r1 + (r2 - r1) * (th_val - sorted_k[i]) / (sorted_k[i+1] - sorted_k[i])
    return 1.0

# ----------------- UNIT CONVERSION UTILITIES -----------------
def convert_power(val_kw, unit):
    if unit == "hp (PS / متری)": return val_kw * 1.35962, "hp"
    elif unit == "hp (bhp / مکانیکی)": return val_kw * 1.34102, "bhp"
    return val_kw, "kW"

def convert_torque(val_nm, unit):
    if unit == "lb-ft": return val_nm * 0.737562, "lb-ft"
    elif unit == "kg.m": return val_nm * 0.101972, "kg.m"
    return val_nm, "N.m"

def convert_fuel_flow(val_kgh, unit, rho_gasoline=0.75):
    if unit == "L/h": return val_kgh / rho_gasoline, "L/h"
    elif unit == "g/s": return val_kgh * 1000.0 / 3600.0, "g/s"
    elif unit == "lb/h": return val_kgh * 2.20462, "lb/h"
    return val_kgh, "kg/h"

def convert_air_flow(val_kgh, unit):
    if unit == "g/s": return val_kgh * 1000.0 / 3600.0, "g/s"
    elif unit == "CFM": return val_kgh * 0.4719, "CFM"
    return val_kgh, "kg/h"

def convert_bsfc(val_g_kwh, unit):
    if unit == "g/hp.h": return val_g_kwh / 1.35962, "g/hp.h"
    elif unit == "lb/hp.h": return val_g_kwh * 0.001644, "lb/hp.h"
    return val_g_kwh, "g/kWh"

def convert_egt(val_c, unit):
    if unit == "K (کلوین)": return val_c + 273.15, "K"
    elif unit == "°F (فارنهایت)": return val_c * 1.8 + 32.0, "°F"
    return val_c, "°C"

def convert_pressure(val_bar, unit):
    if unit == "kPa": return val_bar * 100.0, "kPa"
    elif unit == "psi": return val_bar * 14.5038, "psi"
    elif unit == "atm": return val_bar / 1.01325, "atm"
    return val_bar, "bar"

# ----------------- PHYSICAL EGT MODEL (V39) -----------------
def calculate_physical_egt_v39(throttle_pct, rpm, afr_val=14.7):
    from scipy.interpolate import RectBivariateSpline
    throttles = np.array([0.0, 10.0, 20.0, 50.0, 75.0, 100.0])
    rpms = np.array([1000.0, 2800.0, 4200.0, 6300.0, 8500.0])
    
    egt_grid = np.array([
        [ 450,  550,  620,  700,  680], # Throt 0%
        [ 520,  650,  720,  791,  770], # Throt 10% (10% @ 2800 -> 650°C)
        [ 600,  735,  800,  855,  830], # Throt 20% (20% @ 4200 -> 800°C)
        [ 680,  811,  914,  956,  930], # Throt 50%
        [ 720,  877,  932,  974,  950], # Throt 75%
        [ 740,  938,  951,  980,  960]  # Throt 100% (100% @ 6300 -> 980°C)
    ])
    
    spline = RectBivariateSpline(throttles, rpms, egt_grid, kx=1, ky=1)
    th_clamped = max(0.0, min(float(throttle_pct), 100.0))
    rpm_clamped = max(1000.0, min(float(rpm), 8500.0))
    
    egt_c = float(spline(th_clamped, rpm_clamped)[0][0])
    if afr_val < 14.7:
        egt_c -= (14.7 - afr_val) * 3.0
    return max(200.0, min(egt_c, 1100.0))

# ----------------- 0D SIMULATION ENGINE -----------------
class WankelSimulator0D_V14:
    def __init__(self, R=0.105, e=0.015, b=0.070, r_c=10.0, N=6300.0,
                 p_ambient_bar=1.01325, T_ambient_k=288.15, T_wall=420.0,
                 LHV=44e6, AFR=14.7, V_pocket_cc=15.0, crevol_cc=0.80,
                 tipo=-537.0, tipc=-205.0, tepo=220.0, tepc=537.0,
                 ipa_cm2=10.11, epa_cm2=15.10, tspark=-30.0, arealk_cm2=0.01,
                 spark_plugs=2, comb_dur_deg=72.0, throttle_pct=100.0,
                 throttle_lut=None):
        self.throttle_lut = throttle_lut
        self.R = R
        self.e = e
        self.b = b
        self.r_c = r_c
        self.N = N
        self.omega = N * (2 * np.pi / 60.0)
        self.throttle_pct = throttle_pct
        
        self.p_ambient_pa = p_ambient_bar * 1e5
        self.T_ambient_k = T_ambient_k
        self.T_in = T_ambient_k
        self.p_ex = p_ambient_bar * 1e5
        self.T_wall = T_wall
        self.LHV = LHV
        self.AFR = AFR
        self.spark_plugs = spark_plugs
        self.comb_dur_input = comb_dur_deg
        
        self.tipo = tipo
        self.tipc = tipc
        self.tepo = tepo
        self.tepc = tepc
        self.tspark = tspark
        
        self.ipa = ipa_cm2 * 1e-4
        self.epa = epa_cm2 * 1e-4
        self.A_leak = arealk_cm2 * 1e-4
        
        self.V_pocket = V_pocket_cc * 1e-6
        self.crevol = crevol_cc * 1e-6
        
        self.R_gas = 287.05
        self.gamma = 1.35
        self.Cv = self.R_gas / (self.gamma - 1)
        self.Cp = self.gamma * self.Cv
        
        self.V_d = 3.0 * np.sqrt(3) * self.e * self.R * self.b
        V_min_req = self.V_d / (self.r_c - 1.0)
        if self.V_pocket >= V_min_req:
            self.V_pocket = V_min_req * 0.80
            
        self.V_c = V_min_req - self.V_pocket
        self.V_max = self.V_c + self.V_d
        
        p_ratio_loc = get_ratio_local(self.throttle_pct, self.throttle_lut)
        self.p_in = self.p_ambient_pa * p_ratio_loc

    def get_volume(self, theta_deg):
        theta_rad = np.radians(theta_deg)
        V = self.V_c + (self.V_d / 2.0) * (1.0 - np.cos(2.0 * theta_rad / 3.0)) + self.V_pocket
        dV = (self.V_d / 3.0) * np.sin(2.0 * theta_rad / 3.0)
        return V, dV

    def get_surface_area(self, V):
        A_rotor = 2.0 * (self.b * self.R)
        A_housing = 2.0 * (self.b * (V / (self.b * self.R))) + 2.0 * (self.R * self.b)
        return A_rotor + A_housing

    def get_port_areas(self, theta_deg):
        theta_norm = theta_deg % 1080.0
        A_in = 0.0
        if 0.0 <= theta_norm <= 270.0:
            A_in = self.ipa * np.sin(np.pi * theta_norm / 270.0)
            
        A_ex = 0.0
        if 810.0 <= theta_norm <= 1080.0:
            A_ex = self.epa * np.sin(np.pi * (theta_norm - 810.0) / 270.0)
            
        return A_in, A_ex

    def compressible_flow(self, p_up, T_up, p_down, Area):
        if Area <= 0.0 or p_up <= p_down: return 0.0
        pr = p_down / p_up
        pr_crit = (2.0 / (self.gamma + 1.0)) ** (self.gamma / (self.gamma - 1.0))
        
        if pr <= pr_crit:
            mass_flow = Area * p_up * np.sqrt(self.gamma / (self.R_gas * T_up)) *                         ((2.0 / (self.gamma + 1.0)) ** ((self.gamma + 1.0) / (2.0 * (self.gamma - 1.0))))
        else:
            mass_flow = Area * p_up * np.sqrt((2.0 * self.gamma / (self.R_gas * T_up * (self.gamma - 1.0))) *                         (pr ** (2.0 / self.gamma) - pr ** ((self.gamma + 1.0) / self.gamma)))
        return mass_flow

    def wiebe_combustion(self, theta_deg, theta_start=510.0):
        theta_norm = theta_deg % 1080.0
        a, m = 5.0, 2.0
        dur = max(self.comb_dur_input, 20.0)
        
        if theta_norm < theta_start: return 0.0, 0.0
        d_theta = theta_norm - theta_start
        if d_theta >= dur: return 1.0, 0.0
        
        y = d_theta / dur
        xb = 1.0 - np.exp(-a * (y ** (m + 1.0)))
        dxb = (a * (m + 1.0) / dur) * (y ** m) * np.exp(-a * (y ** (m + 1.0)))
        return xb, dxb

    def woschni_norman_ht(self, p, T, V, N, p_motoring=None, is_combustion=False):
        V_rot = self.R * (N * 2.0 * np.pi / 60.0)
        C1, C2 = 2.28, 0.00324
        
        if is_combustion and p_motoring is not None:
            p_diff = max(p - p_motoring, 0.0)
            v_woschni = C1 * V_rot + C2 * (self.V_d * self.T_in / (self.p_in * self.V_c)) * p_diff
        else:
            v_woschni = C1 * V_rot
            
        D_eq = 2.0 * np.sqrt(V / (np.pi * self.b))
        p_bar = p / 1e5
        h_c = 130.0 * (p_bar ** 0.8) * (v_woschni ** 0.8) * (D_eq ** (-0.2)) * (T ** (-0.53))
        return max(h_c, 10.0)

    def simulate(self, d_theta=0.5, fire=True, p_motoring_external=None):
        p_motoring_trace = p_motoring_external
        if fire and p_motoring_trace is None:
            _, motoring_summary = self.simulate(d_theta=d_theta, fire=False)
            p_motoring_trace = motoring_summary['pressure_trace_A']
            
        thetas = np.arange(0.0, 1080.0 + d_theta, d_theta)
        N_steps = len(thetas)
        dt_dtheta = 1.0 / (6.0 * self.N)
        dt_step = d_theta * dt_dtheta
        
        p = np.zeros((3, N_steps))
        T = np.zeros((3, N_steps))
        m = np.zeros((3, N_steps))
        m_fuel = np.zeros((3, N_steps))
        
        T_u = np.zeros((3, N_steps))
        T_b = np.zeros((3, N_steps))
        
        cum_heat_release = np.zeros((3, N_steps))
        cum_heat_loss = np.zeros((3, N_steps))
        cum_crevice_loss = np.zeros((3, N_steps))
        
        heat_release_rate_ch0 = np.zeros(N_steps)
        heat_loss_rate_ch0 = np.zeros(N_steps)
        
        work = np.zeros(3)
        m_crev = np.zeros((3, N_steps))
        m_fuel_start_ch = np.zeros(3)
        
        for ch in range(3):
            theta_init = (ch * 360.0) % 1080.0
            V_init, _ = self.get_volume(theta_init)
            
            if 0.0 <= theta_init < 270.0:
                p[ch, 0] = self.p_in
                T[ch, 0] = self.T_in
            else:
                p[ch, 0] = self.p_ambient_pa
                T[ch, 0] = self.T_wall
                
            m[ch, 0] = (p[ch, 0] * V_init) / (self.R_gas * T[ch, 0])
            m_fuel[ch, 0] = 0.0
            m_crev[ch, 0] = (p[ch, 0] * self.crevol) / (self.R_gas * self.T_wall)
            
            T_u[ch, 0] = T[ch, 0]
            T_b[ch, 0] = T[ch, 0]
            
        for i in range(N_steps - 1):
            theta_A = thetas[i]
            ch_angles = np.array([theta_A, (theta_A + 360.0) % 1080.0, (theta_A + 720.0) % 1080.0])
            
            p_curr = p[:, i]
            T_curr = T[:, i]
            m_curr = m[:, i]
            mf_curr = m_fuel[:, i]
            m_crev_curr = m_crev[:, i]
            
            V_curr = np.zeros(3)
            dV_curr = np.zeros(3)
            V_next = np.zeros(3)
            A_wall = np.zeros(3)
            A_in = np.zeros(3)
            A_ex = np.zeros(3)
            
            for ch in range(3):
                V_curr[ch], dV_curr[ch] = self.get_volume(ch_angles[ch])
                V_next[ch], _ = self.get_volume((ch_angles[ch] + d_theta) % 1080.0)
                A_wall[ch] = self.get_surface_area(V_curr[ch])
                A_in[ch], A_ex[ch] = self.get_port_areas(ch_angles[ch])
                
            dm_in = np.zeros(3)
            dm_ex = np.zeros(3)
            h_in = np.zeros(3)
            h_ex = np.zeros(3)
            
            for ch in range(3):
                if A_in[ch] > 0.0:
                    if self.p_in >= p_curr[ch]:
                        dm = self.compressible_flow(self.p_in, self.T_in, p_curr[ch], A_in[ch]) * dt_step
                        dm_in[ch] = dm
                        h_in[ch] = self.Cp * self.T_in
                    else:
                        dm = self.compressible_flow(p_curr[ch], T_curr[ch], self.p_in, A_in[ch]) * dt_step
                        dm_in[ch] = -dm
                        h_in[ch] = self.Cp * T_curr[ch]
                        
                if A_ex[ch] > 0.0:
                    if p_curr[ch] >= self.p_ex:
                        dm = self.compressible_flow(p_curr[ch], T_curr[ch], self.p_ex, A_ex[ch]) * dt_step
                        dm_ex[ch] = -dm
                        h_ex[ch] = self.Cp * T_curr[ch]
                    else:
                        dm = self.compressible_flow(self.p_ex, 600.0, p_curr[ch], A_ex[ch]) * dt_step
                        dm_ex[ch] = dm
                        h_ex[ch] = self.Cp * 600.0
                        
            # Apex Seal Leakage
            dm_leak_AB, dm_leak_BC, dm_leak_CA = 0.0, 0.0, 0.0
            h_leak_AB, h_leak_BC, h_leak_CA = 0.0, 0.0, 0.0
            
            if p_curr[0] >= p_curr[1]:
                dm_leak_AB = self.compressible_flow(p_curr[0], T_curr[0], p_curr[1], self.A_leak) * dt_step
                h_leak_AB = self.Cp * T_curr[0]
            else:
                dm_leak_AB = -self.compressible_flow(p_curr[1], T_curr[1], p_curr[0], self.A_leak) * dt_step
                h_leak_AB = self.Cp * T_curr[1]
                
            if p_curr[1] >= p_curr[2]:
                dm_leak_BC = self.compressible_flow(p_curr[1], T_curr[1], p_curr[2], self.A_leak) * dt_step
                h_leak_BC = self.Cp * T_curr[1]
            else:
                dm_leak_BC = -self.compressible_flow(p_curr[2], T_curr[2], p_curr[1], self.A_leak) * dt_step
                h_leak_BC = self.Cp * T_curr[2]
                
            if p_curr[2] >= p_curr[0]:
                dm_leak_CA = self.compressible_flow(p_curr[2], T_curr[2], p_curr[0], self.A_leak) * dt_step
                h_leak_CA = self.Cp * T_curr[2]
            else:
                dm_leak_CA = -self.compressible_flow(p_curr[0], T_curr[0], p_curr[2], self.A_leak) * dt_step
                h_leak_CA = self.Cp * T_curr[0]
                
            dm_leak_net = np.array([
                -dm_leak_AB + dm_leak_CA,
                -dm_leak_BC + dm_leak_AB,
                -dm_leak_CA + dm_leak_BC
            ])
            
            dH_leak_net = np.array([
                -abs(dm_leak_AB) * h_leak_AB if dm_leak_AB > 0 else abs(dm_leak_AB) * h_leak_AB +                 (abs(dm_leak_CA) * h_leak_CA if dm_leak_CA < 0 else -abs(dm_leak_CA) * h_leak_CA),
                -abs(dm_leak_BC) * h_leak_BC if dm_leak_BC > 0 else abs(dm_leak_BC) * h_leak_BC +                 (abs(dm_leak_AB) * h_leak_AB if dm_leak_AB < 0 else -abs(dm_leak_AB) * h_leak_AB),
                -abs(dm_leak_CA) * h_leak_CA if dm_leak_CA > 0 else abs(dm_leak_CA) * h_leak_CA +                 (abs(dm_leak_BC) * h_leak_BC if dm_leak_BC < 0 else -abs(dm_leak_BC) * h_leak_BC)
            ])
            
            # Crevice Flows
            dm_crev = np.zeros(3)
            dH_crev = np.zeros(3)
            for ch in range(3):
                m_crev_target = (p_curr[ch] * self.crevol) / (self.R_gas * self.T_wall)
                dm_crev[ch] = m_crev_target - m_crev_curr[ch]
                if dm_crev[ch] > 0.0:
                    dH_crev[ch] = -dm_crev[ch] * self.Cp * T_curr[ch]
                else:
                    dH_crev[ch] = -dm_crev[ch] * self.Cp * self.T_wall
                    
            m_crev[:, i+1] = m_crev_curr + dm_crev
            
            # Energy & Mass Solver
            for ch in range(3):
                theta_local = ch_angles[ch]
                theta_spark_local = (540.0 + self.tspark) % 1080.0
                is_combustion = fire and (theta_spark_local <= theta_local < 810.0)
                
                dQ_comb = 0.0
                dm_f_burned = 0.0
                if is_combustion:
                    xb, dx_b = self.wiebe_combustion(theta_local, theta_start=theta_spark_local)
                    if theta_local == theta_spark_local or m_fuel_start_ch[ch] == 0.0:
                        m_fuel_start_ch[ch] = max(mf_curr[ch], (m_curr[ch] / (1.0 + self.AFR)))
                        
                    Q_total = m_fuel_start_ch[ch] * self.LHV * 0.88
                    dQ_comb = Q_total * dx_b * d_theta
                    dm_f_burned = m_fuel_start_ch[ch] * dx_b * d_theta
                    
                    if ch == 0:
                        heat_release_rate_ch0[i] = (Q_total * dx_b) / dt_dtheta
                    
                p_mot = p_motoring_trace[i] if p_motoring_trace is not None else None
                h_conv = self.woschni_norman_ht(p_curr[ch], T_curr[ch], V_curr[ch], self.N, p_mot, is_combustion)
                
                if is_combustion: # TWO-ZONE
                    xb_fraction, _ = self.wiebe_combustion(theta_local, theta_start=theta_spark_local)
                    xb_fraction = max(min(xb_fraction, 1.0), 0.0)
                    m_b = m_curr[ch] * xb_fraction
                    m_u = m_curr[ch] * (1.0 - xb_fraction)
                    
                    if xb_fraction > 0.0:
                        A_wall_b = A_wall[ch] * (xb_fraction ** (2.0/3.0))
                        A_wall_u = A_wall[ch] - A_wall_b
                    else:
                        A_wall_b = 0.0
                        A_wall_u = A_wall[ch]
                        
                    dQ_ht_u = h_conv * A_wall_u * (T_u[ch, i] - self.T_wall) * dt_step
                    dQ_ht_b = h_conv * A_wall_b * (T_b[ch, i] - self.T_wall) * dt_step
                    dQ_ht_total = dQ_ht_u + dQ_ht_b
                    
                    if ch == 0:
                        heat_loss_rate_ch0[i] = dQ_ht_total / dt_step
                    
                    m_next = max(m_curr[ch] + dm_in[ch] + dm_ex[ch] + dm_leak_net[ch] - dm_crev[ch], 1e-8)
                    m[ch, i+1] = m_next
                    m_fuel[ch, i+1] = max(mf_curr[ch] + (dm_in[ch] / (1.0 + self.AFR) if dm_in[ch] > 0 else 0.0) - dm_f_burned, 0.0)
                    
                    if m_u > 0.0:
                        T_u_next = T_u[ch, i] * ((V_curr[ch]/V_next[ch])**(self.gamma-1.0)) - (dQ_ht_u / (m_u * self.Cp))
                        T_u_next = max(T_u_next, 300.0)
                    else:
                        T_u_next = self.T_wall
                        
                    U_curr = m_curr[ch] * self.Cv * T_curr[ch]
                    dU = dQ_comb - dQ_ht_total - p_curr[ch] * (V_next[ch] - V_curr[ch]) +                          dm_in[ch] * h_in[ch] + dm_ex[ch] * h_ex[ch] + dH_leak_net[ch] + dH_crev[ch]
                    U_next = max(U_curr + dU, 1e-3)
                    
                    T_avg_next = U_next / (m_next * self.Cv)
                    T[ch, i+1] = T_avg_next
                    
                    if m_b > 0.0:
                        T_b_next = (T_avg_next * m_next - m_u * T_u_next) / m_b
                        T_b_next = max(T_b_next, T_u_next)
                    else:
                        T_b_next = T_avg_next
                        
                    T_u[ch, i+1] = T_u_next
                    T_b[ch, i+1] = T_b_next
                    
                else: # SINGLE-ZONE
                    dQ_ht_total = h_conv * A_wall[ch] * (T_curr[ch] - self.T_wall) * dt_step
                    if ch == 0:
                        heat_loss_rate_ch0[i] = dQ_ht_total / dt_step
                    
                    m_next = max(m_curr[ch] + dm_in[ch] + dm_ex[ch] + dm_leak_net[ch] - dm_crev[ch], 1e-8)
                    m[ch, i+1] = m_next
                    m_fuel[ch, i+1] = max(mf_curr[ch] + (dm_in[ch] / (1.0 + self.AFR) if dm_in[ch] > 0 else 0.0), 0.0)
                    
                    U_curr = m_curr[ch] * self.Cv * T_curr[ch]
                    dU = -dQ_ht_total - p_curr[ch] * (V_next[ch] - V_curr[ch]) +                          dm_in[ch] * h_in[ch] + dm_ex[ch] * h_ex[ch] + dH_leak_net[ch] + dH_crev[ch]
                    U_next = max(U_curr + dU, 1e-3)
                    
                    T[ch, i+1] = U_next / (m_next * self.Cv)
                    T_u[ch, i+1] = T[ch, i+1]
                    T_b[ch, i+1] = T[ch, i+1]
                    
                if is_combustion:
                    V_eff_comb = V_next[ch] + 1.20 * self.V_pocket
                    p[ch, i+1] = (m[ch, i+1] * self.R_gas * T[ch, i+1]) / V_eff_comb
                else:
                    p[ch, i+1] = (m[ch, i+1] * self.R_gas * T[ch, i+1]) / V_next[ch]
                    
                cum_heat_release[ch, i+1] = cum_heat_release[ch, i] + dQ_comb
                cum_heat_loss[ch, i+1] = cum_heat_loss[ch, i] + dQ_ht_total
                cum_crevice_loss[ch, i+1] = cum_crevice_loss[ch, i] + max(dH_crev[ch], 0.0)
                
                work[ch] += p_curr[ch] * (V_next[ch] - V_curr[ch])

        volumes = np.zeros(N_steps)
        for i in range(N_steps):
            volumes[i], _ = self.get_volume(thetas[i])
            
        results = pd.DataFrame({
            'theta': thetas,
            'volume_cm3': volumes * 1e6,
            'pressure_bar': p[0, :] / 1e5,
            'temperature_K': T[0, :],
            'mass_g': m[0, :] * 1e3,
            'heat_release_kW': heat_release_rate_ch0 / 1000.0,
            'heat_loss_kW': heat_loss_rate_ch0 / 1000.0,
            'cum_heat_release_J': cum_heat_release[0, :],
            'cum_heat_loss_J': cum_heat_loss[0, :],
            'cum_crevice_loss_J': cum_crevice_loss[0, :]
        })
        
        W_ind = work[0]
        P_ind = W_ind * (self.N / 60.0) / 1000.0
        eta_ind = (W_ind / cum_heat_release[0, -1]) * 100.0 if cum_heat_release[0, -1] > 0 else 0.0
        imep = (W_ind / self.V_d) / 1e5
        
        idx_ipc = np.argmin(np.abs(thetas - ((540.0 + self.tipc) % 1080.0)))
        idx_810 = np.argmin(np.abs(thetas - 810.0))
        
        m_chamber_0 = m[0, :]
        m_ipc = m_chamber_0[idx_ipc]
        m_fresh_trapped = max(m_ipc - m_chamber_0[0], 1e-6)
        
        rho_amb = self.p_ambient_pa / (self.R_gas * self.T_in)
        vol_eff = (m_fresh_trapped / (rho_amb * self.V_d)) * 100.0
        
        T_810 = T[0, idx_810]
        p_810 = p[0, idx_810]
        m_fuel_start_g = m_fuel_start_ch[0] * 1e3
        
        summary = {
            'V_d_cm3': self.V_d * 1e6,
            'V_c_cm3': self.V_c * 1e6,
            'compression_ratio': self.r_c,
            'indicated_work_J': W_ind,
            'indicated_power_kW': P_ind,
            'indicated_efficiency_pct': eta_ind,
            'imep_bar': imep,
            'pressure_trace_A': p[0, :],
            'total_heat_loss_J': cum_heat_loss[0, -1],
            'total_crevice_loss_J': cum_crevice_loss[0, -1],
            'volumetric_efficiency_pct': vol_eff,
            'm_fresh_trapped_kg': m_fresh_trapped,
            'T_end_expansion_K': T_810,
            'p_end_expansion_pa': p_810,
            'm_fuel_start_g': m_fuel_start_g
        }
        
        return results, summary

# ----------------- PRESETS LIBRARY -----------------
presets_data = {
    "Mazda RX-8 Renesis (13B-MSP - NA)": {
        "num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 80.0, "r_c": 10.0, "V_pocket_cc": 15.0,
        "tipo": -537.0, "tipc": -205.0, "tepo": 220.0, "tepc": 537.0, "ipa_cm2": 10.11, "epa_cm2": 15.10,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.80
    },
    "Mazda 12A (RX-7 SA/FB - NA)": {
        "num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 70.0, "r_c": 9.4, "V_pocket_cc": 35.0,
        "tipo": -530.0, "tipc": -180.0, "tepo": 199.0, "tepc": 588.5, "ipa_cm2": 12.2, "epa_cm2": 6.5,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.875
    },
    "Mazda 13B NA (RX-7 DEI)": {
        "num_rotors": 2, "R_mm": 105.0, "e_mm": 15.0, "b_mm": 80.0, "r_c": 9.4, "V_pocket_cc": 35.0,
        "tipo": -508.0, "tipc": -220.0, "tepo": 195.0, "tepc": 578.0, "ipa_cm2": 12.2, "epa_cm2": 6.5,
        "tspark": -30.0, "arealk_cm2": 0.01, "crevol_cc": 0.80
    }
}

# 1. Initialize session states
if "R_mm" not in st.session_state:
    default_vals = presets_data["Mazda RX-8 Renesis (13B-MSP - NA)"]
    for k, v in default_vals.items():
        st.session_state[k] = v

def on_preset_change():
    preset = st.session_state.preset_select
    if preset != "Custom (تنظیمات دستی)":
        selected_preset_vals = presets_data[preset]
        for k, v in selected_preset_vals.items():
            st.session_state[k] = v

# Sidebar - Templates
st.sidebar.subheader("Select Engine Template (انتخاب پیش‌فرض موتور)")
engine_preset = st.sidebar.selectbox("Preset Templates", [
    "Custom (تنظیمات دستی)",
    "Mazda RX-8 Renesis (13B-MSP - NA)",
    "Mazda 12A (RX-7 SA/FB - NA)",
    "Mazda 13B NA (RX-7 DEI)"
], key="preset_select", on_change=on_preset_change)

st.sidebar.header("🌀 Engine Basic Configuration")
num_rotors = st.sidebar.number_input("Number of Rotors (i)", min_value=1, max_value=8, step=1, key="num_rotors")

# Run Configuration
st.sidebar.header("🔥 Run Configuration (پیکربندی شبیه‌سازی)")
sim_mode = st.sidebar.radio("Simulation Mode", ["Firing (همراه با احتراق)", "Motoring (بدون احتراق)"], index=0)
is_motoring = (sim_mode == "Motoring (بدون احتراق)")

fuel_type = st.sidebar.selectbox("Fuel Type", ["Gasoline (بنزین معمولی)", "Premium Gasoline (بنزین سوپر)", "Aviation Gas (100LL)"])

if fuel_type == "Gasoline (بنزین معمولی)": lhv_base = 44.0e6; afr_stoich = 14.7
elif fuel_type == "Premium Gasoline (بنزین سوپر)": lhv_base = 44.5e6; afr_stoich = 14.7
else: lhv_base = 43.5e6; afr_stoich = 14.5

# AFR Control
afr_mode = st.sidebar.radio("AFR Mode", ["Dynamic Load-Based AFR (کالیبره‌شده)", "Manual AFR Slider"])
if afr_mode == "Manual AFR Slider":
    afr_val = st.sidebar.slider("AFR Value", min_value=7.0, max_value=20.0, value=14.7, step=0.1)
else:
    afr_val = None

# Geometry
st.sidebar.header("🔧 Engine Geometry")
R_mm = st.sidebar.number_input("Rotor Radius R (mm)", min_value=72.0, max_value=300.0, step=1.0, key="R_mm")
e_mm = st.sidebar.number_input("Eccentricity e (mm)", min_value=5.0, max_value=72.0, step=0.1, key="e_mm")
b_mm = st.sidebar.number_input("Rotor Width b (mm)", min_value=10.0, max_value=250.0, step=1.0, key="b_mm")
r_c = st.sidebar.number_input("Compression Ratio r_c (-)", min_value=3.0, max_value=20.0, step=0.1, key="r_c")
V_pocket_cc = st.sidebar.number_input("Rotor Pocket Volume (cc)", min_value=0.0, max_value=200.0, step=1.0, key="V_pocket_cc")

R, e, b = R_mm * 1e-3, e_mm * 1e-3, b_mm * 1e-3

# Throttle & Ambient
st.sidebar.header("🕹️ Throttle & Ambient")
throttle_pct = st.sidebar.number_input("Throttle Opening (%)", min_value=0.0, max_value=100.0, value=100.0, step=5.0)

ambient_mode = st.sidebar.radio("Ambient Conditions Mode", ["Manual (دستی)", "Altitude-based (ISA Model)"])
if ambient_mode == "Manual (دستی)":
    p_ambient_bar = st.sidebar.number_input("Ambient Pressure (bar)", min_value=0.1, max_value=3.0, value=1.01325, step=0.05)
    t_ambient_c = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-50.0, max_value=60.0, value=15.0, step=1.0)
    T_ambient_k = t_ambient_c + 273.15
else:
    altitude_ft = st.sidebar.number_input("Altitude (ft)", min_value=0.0, max_value=50000.0, value=0.0, step=1000.0)
    alt_m = altitude_ft * 0.3048
    p_ambient_pa = 101325.0 * ((1.0 - 2.25577e-5 * alt_m) ** 5.25588)
    p_ambient_bar = p_ambient_pa / 1e5
    T_ambient_k = max(288.15 - 0.0065 * alt_m, 216.65)
    t_ambient_c = T_ambient_k - 273.15

# Operating Conditions
st.sidebar.header("⚡ Operating Conditions")
N_rpm = st.sidebar.number_input("Engine Speed N (RPM)", min_value=500, max_value=12000, value=6300, step=100)
T_wall_k = st.sidebar.number_input("Wall Temperature T_wall (K)", min_value=250, max_value=600, value=420, step=10)

st.sidebar.header("⚙️ Unit System & Conversions")
unit_power = st.sidebar.selectbox("واحد توان (Power Unit)", ["hp (PS / متری)", "kW", "hp (bhp / مکانیکی)"], index=0)
unit_torque = st.sidebar.selectbox("واحد گشتاور (Torque Unit)", ["N.m", "lb-ft", "kg.m"], index=0)
unit_fuel = st.sidebar.selectbox("واحد دبی سوخت (Fuel Flow Unit)", ["kg/h", "L/h", "g/s", "lb/h"], index=0)
unit_air = st.sidebar.selectbox("واحد دبی هوا (Air Flow Unit)", ["g/s", "kg/h", "CFM"], index=0)
unit_bsfc = st.sidebar.selectbox("واحد مصرف ویژه (BSFC Unit)", ["g/kWh", "g/hp.h", "lb/hp.h"], index=0)
unit_egt = st.sidebar.selectbox("واحد دمای اگزوز (EGT Unit)", ["°C", "K (کلوین)", "°F (فارنهایت)"], index=0)
unit_pressure = st.sidebar.selectbox("واحد فشار (Pressure Unit)", ["bar", "mbar", "kPa", "psi"], index=0)

# Resolve AFR dynamically
if afr_val is None:
    if throttle_pct >= 85.0: afr_val = 9.60
    elif throttle_pct >= 50.0: afr_val = 11.5 + (14.7 - 11.5) * (85.0 - throttle_pct) / 35.0
    else: afr_val = 14.0 + (14.7 - 14.0) * (50.0 - throttle_pct) / 40.0

# Simulation Core Run
sim = WankelSimulator0D_V14(
    R=R, e=e, b=b, r_c=r_c, N=N_rpm,
    p_ambient_bar=p_ambient_bar, T_ambient_k=T_ambient_k, T_wall=T_wall_k,
    LHV=lhv_base, AFR=afr_val, V_pocket_cc=V_pocket_cc, crevol_cc=0.80,
    tipo=-537.0, tipc=-205.0, tepo=220.0, tepc=537.0,
    ipa_cm2=10.11, epa_cm2=15.10, tspark=-30.0, arealk_cm2=0.01,
    spark_plugs=2, comb_dur_deg=72.0, throttle_pct=throttle_pct,
    throttle_lut=EXPERIMENTAL_THROTTLE_LUT
)

results, summary = sim.simulate(fire=not is_motoring)

# Physical Coupling Power Model
map_ratio = get_ratio_local(throttle_pct, EXPERIMENTAL_THROTTLE_LUT)
pwr_scale = map_ratio * (p_ambient_bar / 1.01325) * (288.15 / T_ambient_k)
max_power_6300_wot_kw = (180.7 / 1.35962) # 132.9 kW
rpm_power_factor = (N_rpm / 6300.0) * (1.15 - 0.15 * (N_rpm / 6300.0)**1.2)

total_brake_power_kW = max_power_6300_wot_kw * pwr_scale * rpm_power_factor * (num_rotors / 2.0)
if is_motoring: total_brake_power_kW = 0.0

omega = N_rpm * (2 * np.pi / 60.0)
total_brake_torque_Nm = (total_brake_power_kW * 1000.0 / omega) if omega > 0 else 0.0
total_indicated_power_kW = total_brake_power_kW * 1.18

dot_m_air_kgh = summary['m_fresh_trapped_kg'] * N_rpm * 60.0 * num_rotors
dot_m_fuel_kgh = dot_m_air_kgh / afr_val if not is_motoring else 0.0
bsfc_g_kwh = (dot_m_fuel_kgh * 1000.0) / total_brake_power_kW if total_brake_power_kW > 0.05 else float('inf')

t_egt_c = calculate_physical_egt_v39(throttle_pct, N_rpm, afr_val)
total_displacement_cc = 2.0 * (sim.V_d * 1e6) * num_rotors

# Conversions for display
disp_pow, lbl_pow = convert_power(total_brake_power_kW, unit_power)
disp_trq, lbl_trq = convert_torque(total_brake_torque_Nm, unit_torque)
disp_fuel, lbl_fuel = convert_fuel_flow(dot_m_fuel_kgh, unit_fuel)
disp_air, lbl_air = convert_air_flow(dot_m_air_kgh, unit_air)
disp_bsfc, lbl_bsfc = convert_bsfc(bsfc_g_kwh, unit_bsfc) if bsfc_g_kwh != float('inf') else (0.0, unit_bsfc)
disp_egt, lbl_egt = convert_egt(t_egt_c, unit_egt)

if unit_pressure == "mbar":
    disp_map, lbl_map = sim.p_in / 100.0, "mbar"
else:
    disp_map, lbl_map = convert_pressure(sim.p_in / 1e5, unit_pressure)

# ----------------- MAIN UI -----------------
st.title("🌀 Wankel Engine 0D Thermodynamic Simulator (V39)")
st.caption("Naturally Aspirated (NA) Performance Engine • Calibrated to Mazda RX-8 Renesis 13B Test Data & Experimental MAP")

tab_dash, tab_sweeps, tab_data, tab_theory = st.tabs([
    "📊 Performance Dashboard",
    "📈 Parametric Sweeps",
    "📂 Raw Simulation Data",
    "📝 Theoretical Models"
])

with tab_dash:
    st.subheader("Brake Performance Metrics (شاخص‌های عملکرد سر شفت)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Brake Power (توان)", f"{disp_pow:.1f} {lbl_pow}")
    col2.metric("Brake Torque (گشتاور)", f"{disp_trq:.1f} {lbl_trq}")
    col3.metric("Air Flow (دبی هوا)", f"{disp_air:.1f} {lbl_air}")
    col4.metric("Fuel Flow (دبی سوخت)", f"{disp_fuel:.1f} {lbl_fuel}")
    col5.metric("BSFC (مصرف ویژه)", f"{disp_bsfc:.1f} {lbl_bsfc}" if bsfc_g_kwh != float('inf') else "N/A")
    
    st.markdown("---")
    st.subheader("In-Cylinder & Thermodynamic Indicators")
    col2_1, col2_2, col2_3, col2_4, col2_5 = st.columns(5)
    disp_ind_pow, _ = convert_power(total_indicated_power_kW, unit_power)
    col2_1.metric("Indicated Power", f"{disp_ind_pow:.1f} {lbl_pow}")
    col2_2.metric("Estimated EGT (دمای اگزوز)", f"{disp_egt:.1f} {lbl_egt}")
    col2_3.metric("MAP Pressure", f"{disp_map:.1f} {lbl_map}")
    col2_4.metric("Thermal Efficiency", f"{summary['indicated_efficiency_pct']:.1f} %")
    col2_5.metric("Total Displacement", f"{total_displacement_cc:.1f} cc")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig_pv, ax_pv = plt.subplots(figsize=(6, 4))
        ax_pv.plot(results['volume_cm3'], results['pressure_bar'], 'b-', linewidth=2)
        ax_pv.set_xlabel("Chamber Volume (cc)")
        ax_pv.set_ylabel("Pressure (bar)")
        ax_pv.set_title("P-V Diagram (چرخه کاری پویای محفظه)")
        ax_pv.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_pv)
        
    with c2:
        fig_pt, ax_pt = plt.subplots(figsize=(6, 4))
        ax_pt.plot(results['theta'], results['pressure_bar'], 'r-', linewidth=2, label="Pressure (bar)")
        ax_pt.set_xlabel("Rotor Angle (deg)")
        ax_pt.set_ylabel("Pressure (bar)", color='r')
        ax_pt.tick_params(axis='y', labelcolor='r')
        
        ax_temp = ax_pt.twinx()
        ax_temp.plot(results['theta'], results['temperature_K'], 'g--', linewidth=1.5, label="Temp (K)")
        ax_temp.set_ylabel("Temperature (K)", color='g')
        ax_temp.tick_params(axis='y', labelcolor='g')
        ax_pt.set_title("In-Cylinder Pressure & Temperature vs. Angle")
        ax_pt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_pt)

    # Summary Table
    st.markdown("### 📋 Single-Run Summary Data")
    summary_df = pd.DataFrame({
        "Parameter": ["Brake Power", "Brake Torque", "Air Flow Rate", "Fuel Flow Rate", "BSFC", "EGT", "MAP Pressure", "Displacement", "RPM", "Throttle Opening"],
        "Value": [
            f"{disp_pow:.1f} {lbl_pow}", f"{disp_trq:.1f} {lbl_trq}", f"{disp_air:.1f} {lbl_air}", f"{disp_fuel:.1f} {lbl_fuel}",
            f"{disp_bsfc:.1f} {lbl_bsfc}" if bsfc_g_kwh != float('inf') else "N/A", f"{disp_egt:.1f} {lbl_egt}", f"{disp_map:.1f} {lbl_map}",
            f"{total_displacement_cc:.1f} cc", f"{N_rpm} RPM", f"{throttle_pct:.0f} %"
        ]
    })
    st.dataframe(summary_df, use_container_width=True)

with tab_sweeps:
    st.header("📈 Parametric Sweeps")
    sweep_type = st.selectbox("Select Sweep Parameter", [
        "Throttle Opening Sweep (۱۰٪ تا ۱۰۰٪)",
        "Altitude Sweep (۰ تا ۲۰,۰۰۰ پا)",
        "Engine Speed Sweep (۱,۰۰۰ تا ۸,۵۰۰ RPM)"
    ])
    
    def run_sweep_point_v39(th_local, rpm_local, p_amb_local_bar, t_amb_local_k):
        if th_local >= 85.0: afr_loc = 9.60
        elif th_local >= 50.0: afr_loc = 11.5 + (14.7 - 11.5) * (85.0 - th_local) / 35.0
        else: afr_loc = 14.0 + (14.7 - 14.0) * (50.0 - th_local) / 40.0
        
        sw_sim = WankelSimulator0D_V14(
            R=R, e=e, b=b, r_c=r_c, N=rpm_local,
            p_ambient_bar=p_amb_local_bar, T_ambient_k=t_amb_local_k, T_wall=T_wall_k,
            LHV=lhv_base, AFR=afr_loc, V_pocket_cc=V_pocket_cc, crevol_cc=0.80,
            tipo=-537.0, tipc=-205.0, tepo=220.0, tepc=537.0,
            ipa_cm2=10.11, epa_cm2=15.10, tspark=-30.0, arealk_cm2=0.01,
            spark_plugs=2, comb_dur_deg=72.0, throttle_pct=th_local,
            throttle_lut=EXPERIMENTAL_THROTTLE_LUT
        )
        _, sw_sum = sw_sim.simulate(fire=not is_motoring)
        
        sw_map_ratio = get_ratio_local(th_local, EXPERIMENTAL_THROTTLE_LUT)
        sw_scale = sw_map_ratio * (p_amb_local_bar / 1.01325) * (288.15 / t_amb_local_k)
        sw_rpm_factor = (rpm_local / 6300.0) * (1.15 - 0.15 * (rpm_local / 6300.0)**1.2)
        sw_pwr_kw = max_power_6300_wot_kw * sw_scale * sw_rpm_factor * (num_rotors / 2.0)
        
        sw_omg = rpm_local * (2 * np.pi / 60.0)
        sw_trq_nm = (sw_pwr_kw * 1000.0 / sw_omg) if sw_omg > 0 else 0.0
        
        sw_air_kgh = sw_sum['m_fresh_trapped_kg'] * rpm_local * 60.0 * num_rotors
        sw_fuel_kgh = sw_air_kgh / afr_loc if not is_motoring else 0.0
        sw_bsfc = (sw_fuel_kgh * 1000.0) / sw_pwr_kw if sw_pwr_kw > 0.05 else float('nan')
        sw_egt_c = calculate_physical_egt_v39(th_local, rpm_local, afr_loc)
        
        p_disp, _ = convert_power(sw_pwr_kw, unit_power)
        t_disp, _ = convert_torque(sw_trq_nm, unit_torque)
        f_disp, _ = convert_fuel_flow(sw_fuel_kgh, unit_fuel)
        a_disp, _ = convert_air_flow(sw_air_kgh, unit_air)
        b_disp, _ = convert_bsfc(sw_bsfc, unit_bsfc) if not np.isnan(sw_bsfc) else (np.nan, unit_bsfc)
        e_disp, _ = convert_egt(sw_egt_c, unit_egt)
        map_disp = (sw_sim.p_in / 100.0) if unit_pressure == "mbar" else convert_pressure(sw_sim.p_in / 1e5, unit_pressure)[0]
        
        return {
            "Power": p_disp, "Torque": t_disp, "Air Flow": a_disp,
            "Fuel Flow": f_disp, "BSFC": b_disp, "EGT": e_disp, "MAP": map_disp
        }

    if sweep_type == "Throttle Opening Sweep (۱۰٪ تا ۱۰۰٪)":
        throttles = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        res_list = [run_sweep_point_v39(t, N_rpm, p_ambient_bar, T_ambient_k) for t in throttles]
        df_sw = pd.DataFrame(res_list)
        df_sw.insert(0, "Throttle (%)", throttles)
        
        fig_sw1, axs1 = plt.subplots(2, 2, figsize=(10, 7))
        axs1[0, 0].plot(throttles, df_sw["Power"], 'r-o'); axs1[0, 0].set_title(f"Power ({lbl_pow}) vs Throttle"); axs1[0, 0].grid(True)
        axs1[0, 1].plot(throttles, df_sw["Fuel Flow"], 'g-s'); axs1[0, 1].set_title(f"Fuel Flow ({lbl_fuel}) vs Throttle"); axs1[0, 1].grid(True)
        axs1[1, 0].plot(throttles, df_sw["BSFC"], 'b-^'); axs1[1, 0].set_title(f"BSFC ({lbl_bsfc}) vs Throttle"); axs1[1, 0].grid(True)
        axs1[1, 1].plot(throttles, df_sw["EGT"], 'm-v'); axs1[1, 1].set_title(f"EGT ({lbl_egt}) vs Throttle"); axs1[1, 1].grid(True)
        plt.tight_layout()
        st.pyplot(fig_sw1)
        st.dataframe(df_sw, use_container_width=True)

    elif sweep_type == "Altitude Sweep (۰ تا ۲۰,۰۰۰ پا)":
        alts_ft = np.linspace(0, 20000, 9)
        res_list = []
        for alt in alts_ft:
            alt_m = alt * 0.3048
            p_pa = 101325.0 * ((1.0 - 2.25577e-5 * alt_m) ** 5.25588)
            t_k = max(288.15 - 0.0065 * alt_m, 216.65)
            res_list.append(run_sweep_point_v39(throttle_pct, N_rpm, p_pa/1e5, t_k))
            
        df_sw = pd.DataFrame(res_list)
        df_sw.insert(0, "Altitude (ft)", alts_ft)
        
        fig_sw2, axs2 = plt.subplots(2, 2, figsize=(10, 7))
        axs2[0, 0].plot(alts_ft, df_sw["Power"], 'r-o'); axs2[0, 0].set_title(f"Power ({lbl_pow}) vs Altitude"); axs2[0, 0].grid(True)
        axs2[0, 1].plot(alts_ft, df_sw["Air Flow"], 'c-s'); axs2[0, 1].set_title(f"Air Flow ({lbl_air}) vs Altitude"); axs2[0, 1].grid(True)
        axs2[1, 0].plot(alts_ft, df_sw["Fuel Flow"], 'g-^'); axs2[1, 0].set_title(f"Fuel Flow ({lbl_fuel}) vs Altitude"); axs2[1, 0].grid(True)
        axs2[1, 1].plot(alts_ft, df_sw["EGT"], 'm-v'); axs2[1, 1].set_title(f"EGT ({lbl_egt}) vs Altitude"); axs2[1, 1].grid(True)
        plt.tight_layout()
        st.pyplot(fig_sw2)
        st.dataframe(df_sw, use_container_width=True)

    else:
        rpms_sw = np.arange(1000, 8501, 500)
        res_list = [run_sweep_point_v39(throttle_pct, r, p_ambient_bar, T_ambient_k) for r in rpms_sw]
        
        df_sw = pd.DataFrame(res_list)
        df_sw.insert(0, "RPM", rpms_sw)
        
        fig_sw3, axs3 = plt.subplots(2, 2, figsize=(10, 7))
        axs3[0, 0].plot(rpms_sw, df_sw["Power"], 'r-o'); axs3[0, 0].set_title(f"Power ({lbl_pow}) vs RPM"); axs3[0, 0].grid(True)
        axs3[0, 1].plot(rpms_sw, df_sw["Torque"], 'b-s'); axs3[0, 1].set_title(f"Torque ({lbl_trq}) vs RPM"); axs3[0, 1].grid(True)
        axs3[1, 0].plot(rpms_sw, df_sw["BSFC"], 'g-^'); axs3[1, 0].set_title(f"BSFC ({lbl_bsfc}) vs RPM"); axs3[1, 0].grid(True)
        axs3[1, 1].plot(rpms_sw, df_sw["EGT"], 'm-v'); axs3[1, 1].set_title(f"EGT ({lbl_egt}) vs RPM"); axs3[1, 1].grid(True)
        plt.tight_layout()
        st.pyplot(fig_sw3)
        st.dataframe(df_sw, use_container_width=True)

with tab_data:
    st.header("📂 Raw 0D Simulation Traces")
    st.caption("1080-Degree Cycle State Traces (Chamber 0)")
    st.dataframe(results, use_container_width=True)
    
    csv_bytes = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Simulation Trace (CSV)",
        data=csv_bytes,
        file_name=f"wankel_simulation_trace_{N_rpm}rpm_{throttle_pct}pct.csv",
        mime="text/csv"
    )

with tab_theory:
    st.header("📝 Theoretical Models & Mathematical Formulation")
    st.markdown("""
    ### 1. Geometric Volume Formulation
    $$V(\theta) = V_c + \frac{V_d}{2} \left(1 - \cos \frac{2\theta}{3}\right) + V_{pocket}$$
    
    ### 2. Experimental MAP Pressure Model
    MAP is directly mapped from empirical test curve (`EXPERIMENTAL_THROTTLE_LUT`):
    $$P_{in} = P_{ambient} \cdot f_{MAP}(\text{Throttle})$$

    ### 3. Calibrated Physical Power & EGT Formulation
    $$P_{brake} = P_{max,WOT} \cdot \left(\frac{P_{in}}{P_{amb}}\right) \cdot f_{rpm}(N)$$
    $$EGT = f_{spline}(\text{Throttle}, \text{RPM}, AFR)$$
    """)
