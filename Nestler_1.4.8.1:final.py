# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
from functools import partial
from mpl_toolkits.mplot3d import Axes3D

# ======================= Constants and Global Variables =======================

# Define constants for equilibrium calculation
a1, a2 = 1.183e4, -29.061
b1, b2 = -4.773e3, 4.672
#a1, a2, a3, a4, a5, a6, a7 = 7.44140e4, 1.89260e2, 3.2443e-2, 7.0432e-6, -5.6053e-9, 1.0344e-12, -6.4364e1
#b1, b2, b3, b4, b5, b6, b7 = -3.94121e4, -5.41516e1, -5.5642e-2, 2.5760e-5, -7.6594e-9, 1.0161e-12, 1.8429e1

R = 8.314  # Gas constant in J/mol·K

# Parameters for reactor model
reactor_length = 7.002  # m
reactor_diameter = 0.038  # m
A_cross = np.pi * (reactor_diameter/2)**2  # m²
reactor_perimeter = np.pi * reactor_diameter  # m
V_r = A_cross * reactor_length  # reactor volume in m³
rb = 1132  # bulk density (kg/m³)
m_cat = rb * V_r  # catalyst mass (kg)

# Heat transfer parameters
U = 143.0  # Overall heat transfer coefficient (W/m²·K)
#T_wall = 523.15  # Wall temperature (K) - constant

# Reaction enthalpies (kJ/mol)
dH_MeOH = -49.5  # CO2 + 3H2 -> CH3OH + H2O
dH_RWGS = 41.2   # CO2 + H2 -> CO + H2O
dH_MeOH_CO = -90.7  # CO + 2H2 -> CH3OH

# Specific heat capacities (J/mol·K)
Cp = {
    'CO2': 37.1,
    'CO': 29.1,
    'H2': 28.8,
    'MeOH': 43.9,
    'H2O': 33.6
}
# ======================= Helper Functions =======================

def Keq_MeOH_CO(T):
    """Calculate equilibrium constant for MeOH from CO"""
    ln_Keq_bar_inv_sq = a1/T + a2
    #ln_Keq_bar_inv_sq = (1/(R*T)) * (a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4 + a6*T**5 + a7*T*np.log(T))
    Keq_in_bar_inv_sq = np.exp(ln_Keq_bar_inv_sq)
    # Umrechnung von bar⁻² zu Pa⁻² (1 bar = 1e5 Pa)
    return Keq_in_bar_inv_sq / (1e5**2)

def Keq_RWGS(T):
    """Calculate equilibrium constant for RWGS reaction"""
    ln_Keq_RWGS = b1/T + b2
    # ln_Keq_RWGS = (1/(R*T)) * (b1 + b2 * T + b3* T**2 + b4*T**3 + b5*T**4 + b6*T**5 + b7 * T * np.log(T))
    return np.exp(ln_Keq_RWGS)

def Keq_MeOH(T):
    """Calculate equilibrium constant for MeOH from CO2"""
    return Keq_MeOH_CO(T) * Keq_RWGS(T)

def arrhenius(A, Ea, T):
    """Calculate rate constant using Arrhenius equation"""
    return A * np.exp(-Ea / (R * T))
    
def vanthoff(B, dH, T):
    """Calculate adsorption constant using van't Hoff equation"""
    return B * np.exp(-dH / (R * T))

# ======================= Rate Model Function =======================

def rate_model(P_CO2, P_CO, P_H2, P_MeOH, P_H2O, T):
    
    k1 = 5.411e-4 * np.exp(-45458 / (R * T))    # k1 for MeOH formation     #[mol/kg*s*Pa]
    k2 = 24.701 * np.exp(-54970 / (R * T))     #k2 fpr RWGS                #[mol/kg*s*Pa**0.5]
    K1 = 3.321e-18 * np.exp(109959 / (R * T))     # K1 for CO2 adsorption     #[Pa^-1]
    K2 = 8.262e-6                               #K2 for H2 adsorption       #[Pa^-1]
    K3 = 6.430e-14 * np.exp(119570 / (R * T))   # K3 for H2o adsorption     #[Pa^-0.5]
    
    # Calculate equilibrium constants
    Keq_m = Keq_MeOH(T)
    Keq_rwgs =Keq_RWGS(T)
    
    # Calculate driving forces (reaction numerators)
  #  df_MeOH = (P_CO2 * P_H2 - (P_MeOH * P_H2O / (P_H2**2 * Keq_m)))
  #  df_RWGS = (P_CO2 * P_H2**0.5 - (P_CO * P_H2O / (P_H2**0.5 * Keq_rwgs)))
  #  df_MeOH_CO = (P_CO * P_H2**0.5 - (P_MeOH / (P_H2**1.5 * Keq_m_co)))
    EQ1 = (1- (P_MeOH * P_H2O) / (P_CO2 * P_H2**3 * Keq_m))        
    EQ2 = (1- (P_CO * P_H2O) / (P_CO2 * P_H2 * Keq_rwgs))  

    # Denominator for all reactions (LHHW formalism)
  #  term1 = np.maximum(1 + K_CO * P_CO + K1 * P_CO2, epsilon)
    term1 = (1 + K1 * P_CO + K2 * P_CO2)
    term2 = (P_H2**0.5 + K3 * P_H2O)
    denom = term1 * term2
    numerator_MeOH =(k1 * K2 * P_CO2 * P_H2**1.5 * EQ1)
    numerator_RWGS = (k2 * K2 * P_CO2 * P_H2 * EQ2)
    r_test = numerator_MeOH /denom
    
    
    # Calculate rates with limits for numerical stability
    r_MeOH = k1 * K2 * P_CO2 * P_H2**1.5 * EQ1 / denom
    r_RWGS = k2 * K2 * P_CO2 * P_H2 * EQ2 / denom

    return r_MeOH, r_RWGS, k1, k2, K1, K2, K3, Keq_m, Keq_rwgs, term1, term2, denom, EQ1, EQ2, numerator_MeOH, numerator_RWGS

# ======================= Reactor Model Function with Energy Balance =======================

def plug_flow_reactor(z, state, u_s_initial,T_feed, C_total_initial, T_wall):
    """
    ODE function for plug flow reactor model with energy balance
    
    Parameters:
    z - axial position (m)
    state - state vector [C_CO2, C_CO, C_H2, C_MeOH, C_H2O, T]
    u_s_initial - initial gas velocity (m/s)
    params - kinetic parameters
    
    Returns:
    dstate_dz - derivatives [dC_CO2/dz, dC_CO/dz, dC_H2/dz, dC_MeOH/dz, dC_H2O/dz, dT/dz]
    """
    # Set minimum concentration threshold
    min_conc = 1e-20
    
    # Extract concentrations and temperature from state vector
    C = state[:-1]  # Concentrations [mol/m³]
    T = state[-1]   # Temperature [K]
    
    # Ensure positive concentrations
    C = np.array(C, dtype=float)
    C = np.maximum(C, min_conc)
    
    # Calculate partial pressures (in Pa)
    RT = R * T
    P = np.maximum(C * RT, min_conc)

    # Get reaction rates
    r_MeOH, r_RWGS, *_ = rate_model(
        P[0],  # P_CO2
        P[1],  # P_CO
        P[2],  # P_H2
        P[3],  # P_MeOH
        P[4],  # P_H2O
        T
    )
    
    # Calculate local gas velocity considering both stoichiometric and thermal effects
    C_total_local = np.sum(C) #lokale gesamtkonzentration über state vektor berechnen
    C_total_local = max(C_total_local, min_conc) 
    u_s_local = u_s_initial * ( C_total_local / C_total_initial) * (T / T_feed)

    # Calculate reaction heat generation rates (J/m³·s)
    # Convert from kJ/mol to J/mol
    q_MeOH = rb * r_MeOH * -dH_MeOH * 1000
    q_RWGS = rb * r_RWGS * -dH_RWGS * 1000
    
    # Total heat generation rate (J/m³·s)
    q_rxn = q_MeOH + q_RWGS
    
    # Heat transfer to wall (J/m³·s)
    # q_wall = U * A / V * (T_wall - T)
    # A/V = perimeter/area = 4/diameter for circular tube
    q_wall = U * reactor_perimeter / A_cross * (T_wall - T)
    
    # Calculate heat capacity of mixture (J/m³·K)
    Cp_mix = (C[0] * Cp['CO2'] + C[1] * Cp['CO'] + C[2] * Cp['H2'] + 
              C[3] * Cp['MeOH'] + C[4] * Cp['H2O']) / np.sum(C)
    
    # Volumetric heat capacity (J/m³·K)
    rho_Cp = C_total_local * Cp_mix
    
    # Calculate mass balance terms for each component
    # [CO2, CO, H2, MeOH, H2O]
    reaction_terms = np.array([
        rb * (-r_MeOH - r_RWGS),     # CO2
        rb * (r_RWGS),               # CO
        rb * (-3*r_MeOH - r_RWGS),   # H2
        rb * (r_MeOH),               # MeOH
        rb * (r_MeOH + r_RWGS)       # H2O
    ])
    
    # Calculate derivatives with safety checks
    dCdz = np.clip(reaction_terms / u_s_local, -1e7, 1e7)
    
    # Energy balance: dT/dz = (q_rxn + q_wall) / (u_s_local * rho_Cp)
    dTdz = (q_rxn + q_wall) / (u_s_local * rho_Cp)
    dTdz = np.clip(dTdz, -1000, 1000)
    
    # Combine mass and energy balance
    dstate_dz = np.concatenate([dCdz, [dTdz]])
    
    return dstate_dz
# ======================= Simulation Function =======================

def simulate_reactor(GHSV, u_s_initial, T_feed, p_total, y_CO2, y_H2, y_CO, y_MeOH, y_H2O, T_wall):
    """Run reactor simulation and save results"""
    
    P_CO2_i = y_CO2 * p_total
    P_H2_i = y_H2 * p_total
    P_CO_i = y_CO * p_total
    P_MeOH_i = y_MeOH * p_total
    P_H2O_i = y_H2O * p_total

    # Initial conditions (concentrations in mol/m³)
    # [CO2, CO, H2, MeOH, H2O]
    initial_concentrations = np.array([
        P_CO2_i  / (R * T_feed),
        P_CO_i / (R * T_feed),
        P_H2_i / (R * T_feed),
        P_MeOH_i / (R * T_feed),
        P_H2O_i / (R * T_feed)
    ])
    
    # Initial state vector [CO2, CO, H2, MeOH, H2O, T]
    initial_state = np.append(initial_concentrations, T_feed)
    
    # Set up evaluation points
    z_eval = np.linspace(0, reactor_length, 100)
    

    C_total_initial = np.sum(initial_concentrations)
    
    # Create partial function for ODE solver
    partial_reactor = partial(
        plug_flow_reactor,
        u_s_initial=u_s_initial,
        T_feed = T_feed,
        T_wall = T_wall,
        C_total_initial = C_total_initial
    )
    
    # Solve ODE
    sol = solve_ivp(
        partial_reactor,
        [0, reactor_length],
        initial_state,
        method='BDF',
        t_eval=z_eval,
        rtol=1e-6,
        atol=1e-8,
        max_step=reactor_length/20
    )
    
    if not sol.success:
        print("Error: ODE solution failed!")
        return None
    
    # Extract solution
    C_profiles = sol.y[:-1]  # Concentrations [mol/m³]
    T_profile = sol.y[-1]    # Temperature [K]
    z_positions = sol.t      # Axial positions [m]
    
    # Calculate partial pressures and other variables along the reactor
    results = {
        'z': z_positions,
        'T': T_profile,
        'C_CO2': C_profiles[0],
        'C_CO': C_profiles[1],
        'C_H2': C_profiles[2],
        'C_MeOH': C_profiles[3],
        'C_H2O': C_profiles[4],
        'P_CO2': np.zeros_like(z_positions),
        'P_CO': np.zeros_like(z_positions),
        'P_H2': np.zeros_like(z_positions),
        'P_MeOH': np.zeros_like(z_positions),
        'P_H2O': np.zeros_like(z_positions),
        'r_MeOH': np.zeros_like(z_positions),
        'r_RWGS': np.zeros_like(z_positions),
        #'r_MeOH_CO': np.zeros_like(z_positions),
        'dCdz_CO2': np.zeros_like(z_positions),
        'dCdz_CO': np.zeros_like(z_positions),
        'dCdz_H2': np.zeros_like(z_positions),
        'dCdz_MeOH': np.zeros_like(z_positions),
        'dCdz_H2O': np.zeros_like(z_positions),
        'dTdz': np.zeros_like(z_positions),
        'velocity': np.zeros_like(z_positions),
        'q_rxn': np.zeros_like(z_positions),
        'q_wall': np.zeros_like(z_positions),
        'term1': np.zeros_like(z_positions),
        'term2': np.zeros_like(z_positions),
        'denom': np.zeros_like(z_positions),
        'EQ1': np.zeros_like(z_positions), 
        'EQ2': np.zeros_like(z_positions),    
        
    }
    
    # Calculate pressures, reaction rates, and derivatives at each position
    for i, z in enumerate(z_positions):
        # Extract concentrations and temperature at this position
        C = C_profiles[:, i]
        T = T_profile[i]
        
        # Calculate partial pressures (Pa)
        RT = R * T
        P = np.maximum(C * RT, 1e-20)
        p_operation = np.sum(P)

        # Store pressures
        results['P_CO2'][i] = P[0]
        results['P_CO'][i] = P[1]
        results['P_H2'][i] = P[2]
        results['P_MeOH'][i] = P[3]
        results['P_H2O'][i] = P[4]
        
        # Calculate reaction rates
        r_MeOH, r_RWGS, _, _, _, _, _, _, _, term1, term2, denom, EQ1, EQ2, _, _ = rate_model(
            P[0], P[1], P[2], P[3], P[4],
            T
        )
        
        # Store reaction rates
        results['r_MeOH'][i] = r_MeOH
        results['r_RWGS'][i] = r_RWGS
        results['term1'][i] = term1
        results['term2'][i] = term2
        results['denom'][i] = denom
        results['EQ1'][i] = EQ1
        results['EQ2'][i] = EQ2
        
        # Calculate heat generation (kW/m³)
        q_MeOH = rb * r_MeOH * dH_MeOH
        q_RWGS = rb * r_RWGS * dH_RWGS
        q_rxn = q_MeOH + q_RWGS
        
        # Heat transfer to wall (kW/m³)
        q_wall = U * reactor_perimeter / A_cross * (T_wall - T) / 1000
        
        # Store heat terms
        results['q_rxn'][i] = q_rxn
        results['q_wall'][i] = q_wall
        
        # Get state derivatives
        current_state = np.append(C, T)
        derivatives = plug_flow_reactor(z, current_state, u_s_initial,T_feed, C_total_initial, T_wall)
        
        # Store derivatives
        results['dCdz_CO2'][i] = derivatives[0]
        results['dCdz_CO'][i] = derivatives[1]
        results['dCdz_H2'][i] = derivatives[2]
        results['dCdz_MeOH'][i] = derivatives[3]
        results['dCdz_H2O'][i] = derivatives[4]
        results['dTdz'][i] = derivatives[5]
        
        # Calculate gas velocity at this position
        C_total_initial = np.sum(initial_concentrations)
        C_total_local = np.sum(C)
        C_total_local = max(C_total_local, 1e-20) # Sicherheits-Check

        # Lokale Geschwindigkeit unter Berücksichtigung von Molzahl UND Temperatur
        # HINWEIS: "T" ist die lokale Temperatur an der Stelle i
        # gas_velocity ist hier u_s_initial
        results['velocity'][i] = u_s_initial * (C_total_local / C_total_initial) * (T / T_feed)
    
    # Calculate conversion and yield
    initial_P_CO2 = results['P_CO2'][0]
    initial_C_CO2 = initial_concentrations[0] 
    initial_C_CO = initial_concentrations[1] 
    results['X_CO2'] = (initial_P_CO2 - results['P_CO2']) / initial_P_CO2 * 100  # Conversion (%)
    results['Y_MeOH'] = (results['P_MeOH'] / (R * results['T'])) / (initial_P_CO2 / (R * T_feed)) * 100 

    results['Y_MeOH_C'] = results['C_MeOH'] / (initial_C_CO2 + initial_C_CO) * 100 
    


    # results['Y_MeOH'] = results['P_MeOH'] / initial_P_CO2 * 100  # Yield (%)
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_filename = f"Nestler_{int(GHSV)}_1h_{int(T_feed)}_K_{int(p_total/1e5)}bar.csv"
    df.to_csv( output_filename, sep=';', decimal='.', index=False, encoding='utf-8')
    print(f"Reactor profile saved to {output_filename}")
    
    return df

# ======================= Plotting Function =======================

def plot_reactor_profile(df, T_wall):
    """Create plots of the reactor profile"""
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
    
    # Plot 1: Partial Pressures
    ax1.plot(df['z'], df['P_CO2'], 'b-', label='CO₂')
    ax1.plot(df['z'], df['P_CO'], 'r-', label='CO')
    ax1.plot(df['z'], df['P_H2'], 'g-', label='H₂')
    ax1.plot(df['z'], df['P_MeOH'], 'm-', label='MeOH')
    ax1.plot(df['z'], df['P_H2O'], 'c-', label='H₂O')
    ax1.set_xlabel('Reactor Position (m)')
    ax1.set_ylabel('Partial Pressure (Pa)')
    ax1.set_title('Partial Pressures Along Reactor')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Reaction Rates
    ax2.plot(df['z'], df['r_MeOH'], 'b-', label='r_MeOH')
    ax2.plot(df['z'], df['r_RWGS'], 'r-', label='r_RWGS')
    ax2.set_xlabel('Reactor Position (m)')
    ax2.set_ylabel('Reaction Rate (mol/kg_cat/s)')
    ax2.set_title('Reaction Rates Along Reactor')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Temperature Profile
    ax3.plot(df['z'], df['T'], 'r-', linewidth=2)
    ax3.axhline(y=T_wall, color='k', linestyle='--', label=f'Wall Temp ({T_wall} K)')
    ax3.set_xlabel('Reactor Position (m)')
    ax3.set_ylabel('Temperature (K)')
    ax3.set_title('Temperature Profile Along Reactor')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Conversion and Yield
    ax4.plot(df['z'], df['X_CO2'], 'b-', label='CO₂ Conversion')
    ax4.plot(df['z'], df['Y_MeOH'], 'r-', label='MeOH Yield')
    ax4.set_xlabel('Reactor Position (m)')
    ax4.set_ylabel('Percent (%)')
    ax4.set_title('Conversion and Yield Along Reactor')
    ax4.legend()
    ax4.grid(True)


# ======================= Main Function =======================

def main():
    """Main function to run simulation and generate plots"""
    print("Starting reactor simulation with energy balance...")
    
    # Verschiedene GHSV-Werte Temperaturen und Drücke testen
    T_feed_list_K = [423.15, 433.15, 443.15, 453.15, 463.15, 473.15, 483.15, 493.15, 503.15, 513.15, 523.15, 533.15, 543.15, 553.15, 563.15, 573.15, 583.15, 593.15, 603.15, 613.15, 623.15]  # in Kelvin (150-350°C in 10K Schritten)
    GHSV_list = [7800]  # in 1/h (Auf einen Wert gesetzt, um T zu isolieren)
    p_total_list = [ 50e5]  # in Pa (50 bar)
    

    # Gaszusammensetzung:
    min_val = 1e-20
    y_CO2 = 0.25
    y_H2 = 0.75
    y_CO = min_val
    y_MeOH = min_val
    y_H2O = min_val
    y_H2 = y_H2 - 3*min_val


    summary_results = []

    for p_total in p_total_list:

        # Berechne initiale Partialdrücke
        P_CO2_i = y_CO2 * p_total
        P_H2_i = y_H2 * p_total
        P_CO_i = y_CO * p_total
        P_MeOH_i = y_MeOH * p_total
        P_H2O_i = y_H2O * p_total

        print(f"\n>>> Simulation bei Gesamt-Druck: {p_total} bar (P_total = {p_total:.2e} Pa)")
        print(f"Feed composition: CO2={P_CO2_i:.2e} Pa, H2={P_H2_i:.2e} Pa")    

        for T_feed in T_feed_list_K:
            print(f"\nRunning simulations for T_feed = {T_feed} K")
            T_wall = T_feed
            for GHSV in GHSV_list:
                print(f"\nRunning simulation for GHSV = {GHSV} 1/h and T_feed = {T_feed} K")
                # Calculate initial total pressure (Pa)
                initial_concentrations = np.array([
                P_CO2_i  / (R * T_feed),
                P_CO_i / (R * T_feed),
                P_H2_i / (R * T_feed),
                P_MeOH_i / (R * T_feed),
                P_H2O_i / (R * T_feed)
                ])
                RT_feed = R * T_feed
                P_initial = initial_concentrations * RT_feed
                p_operation = np.sum(P_initial)
                porosity = 0.39
                T_ref = 273.15
                p_ref = 101325

                u_s_norm = GHSV * reactor_length * (1- porosity) * 1/3600
                u_s_initial = u_s_norm * (p_ref / p_operation) * (T_feed / T_ref)
                print(f"u_s_initial für GHSV={GHSV}: {u_s_initial:.4f} m/s")
                print(f"  V_r (reactor volume): {V_r:.6f} m³")
                print(f"  A_cross (cross-sectional area): {A_cross:.6f} m²")
                print(f"  T_feed: {T_feed:.2f} K")
                print(f"  p_operation (operating pressure): {p_operation:.2f} Pa")

                
                # Berechne kinetische und Gleichgewichtswerte am Feed
                r_MeOH, r_RWGS, k1, k2, K1, K2, K3, Keq_m, Keq_rwgs, term1, term2, denom, EQ1, EQ2, numerator_MeOH, numerator_RWGS = rate_model(
                    P_CO2_i, P_CO_i, P_H2_i, P_MeOH_i, P_H2O_i,
                    T_feed
                )

                df = simulate_reactor(GHSV, u_s_initial, T_feed, p_total, y_CO2, y_H2, y_CO, y_MeOH, y_H2O, T_wall)

        
        
                if df is not None:
                    # Generate plots
                    #plot_reactor_profile(df)

                    exit_idx = -1  # Last position

                    # benötigte Variablen definieren
                    T_out = df['T'].iloc[exit_idx]
                    T_max = df['T'].max()
                    X_CO2 = df['X_CO2'].iloc[exit_idx]
                    Y_MeOH = df['Y_MeOH'].iloc[exit_idx]
                    q_rxn_max = df['q_rxn'].max()
                    q_wall_max = df['q_wall'].max()

                    print("\nReactor Exit Conditions:")
                    print(f"Temperature: {T_out:.1f} K")
                    print(f"CO2 conversion: {X_CO2:.2f}%")
                    print(f"MeOH yield: {Y_MeOH:.2f}%")
                    print(f"Partial pressures (bar): CO2={df['P_CO2'].iloc[exit_idx]:.3f}, "
                        f"CO={df['P_CO'].iloc[exit_idx]:.3f}, H2={df['P_H2'].iloc[exit_idx]:.3f}, "
                        f"MeOH={df['P_MeOH'].iloc[exit_idx]:.3f}, H2O={df['P_H2O'].iloc[exit_idx]:.3f}")

                    print("\nHeat Information:")
                    print(f"Total reaction heat: {np.mean(df['q_rxn']):.2f} kW/m³")
                    print(f"Maximum temperature: {T_max:.1f} K at position {df['z'].iloc[df['T'].   argmax()]:.3f} m")

                    total_heat_rxn = np.trapezoid(df['q_rxn'], df['z'])
                    total_heat_wall = np.trapezoid(df['q_wall'], df['z'])
                    print(f"Net heat generated by reactions: {total_heat_rxn:.2f} kW/m")
                    print(f"Net heat transferred to wall: {total_heat_wall:.2f} kW/m")
                    print(f"Heat balance closure: {100 * (1 - (total_heat_rxn + total_heat_wall) /  total_heat_rxn):.2f}%")

                    summary_results.append({
                        'GHSV': GHSV,
                        'u_feed': u_s_initial,
                        'u_exit': df['velocity'].iloc[exit_idx],
                        'T_feed': T_feed,
                        'p_total': p_total,
                        'r_MeOH' : r_MeOH,
                        'r_RWGS': r_RWGS,
                        'k1': k1,  
                        'k2': k2,
                        'K1': K1,
                        'K2': K2,
                        'K3': K3,
                        'Keq_m': Keq_m,
                        'Keq_rwgs': Keq_rwgs,
                        'term1' : term1,
                        'term2' : term2,
                        'denom': denom,
                        'EQ1': EQ1,
                        'EQ2': EQ2,
                        'numerator_MeOH': numerator_MeOH,
                        'numerator_RWGS': numerator_RWGS,
                        'C_CO2_feed': P_CO2_i / (R * T_feed),
                        'C_CO_feed': P_CO_i / (R * T_feed),
                        'C_H2_feed': P_H2_i / (R * T_feed),
                        'C_MeOH_feed': P_MeOH_i / (R * T_feed),
                        'C_H2O_feed': P_H2O_i / (R * T_feed),
                        'C_CO2_exit': df['C_CO2'].iloc[exit_idx],
                        'C_CO_exit': df['C_CO'].iloc[exit_idx],
                        'C_H2_exit': df['C_H2'].iloc[exit_idx],
                        'C_MeOH_exit': df['C_MeOH'].iloc[exit_idx],
                        'C_H2O_exit': df['C_H2O'].iloc[exit_idx],
                        'P_MeOH': df['P_MeOH'].iloc[exit_idx],
                        'CO2_conversion_%': X_CO2,
                        'MeOH_yield_%': Y_MeOH,
                        #'q_rxn_max_kW/m3': q_rxn_max,
                        #'q_wall_max_kW/m3': q_wall_max,
                        'P_MeOH_exit': df['P_MeOH'].iloc[exit_idx],
                        'P_H2_exit': df['P_H2'].iloc[exit_idx],
                        'P_CO_exit': df['P_CO'].iloc[exit_idx],
                        'P_CO2_exit': df['P_CO2'].iloc[exit_idx],
                        'P_H2O_exit': df['P_H2O'].iloc[exit_idx],
                        'r_MeOH_exit': df['r_MeOH'].iloc[exit_idx],
                        'r_RWGS_exit': df['r_RWGS'].iloc[exit_idx],
                        
                    })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv("Nestler_summary.csv", sep=';', decimal='.', index=False, encoding='utf-8')
    print("\nSummary CSV saved as Nestler_summary.csv")
    print(summary_df)


if __name__ == "__main__":
    main()
  #  plot_intrinsic_rates_vs_temperature()
