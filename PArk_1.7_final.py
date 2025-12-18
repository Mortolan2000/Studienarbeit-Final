# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

# ======================= Constants and Global Variables =======================

# Define constants for equilibrium calculation
a1, a2 = 1.183e4, -29.061
b1, b2 = -4.773e3, 4.672

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

# Define operating conditions
#T_feed = 473.15  # Feed temperature (K) (200°C)
#gas_velocity = 1.0 # m/s

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
    ln_Keq2 = a1/T + a2
    return np.exp(ln_Keq2)

def Keq_RWGS(T):
    """Calculate equilibrium constant for RWGS reaction"""
    ln_Keq_RWGS = b1/T + b2
    return np.exp(ln_Keq_RWGS)

def Keq_MeOH(T):
    """Calculate equilibrium constant for MeOH from CO2"""
    return Keq_MeOH_CO(T) * Keq_RWGS(T)

# ======================= Rate Model Function =======================

def rate_model(P_CO2, P_CO, P_H2, P_MeOH, P_H2O, T):
    """Calculate reaction rates for CO2 hydrogenation reactions"""
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-30
    
    # Calculate equilibrium constants
    Keq_m = Keq_MeOH(T)
    Keq_rwgs = Keq_RWGS(T)
    Keq_m_co = Keq_MeOH_CO(T)

    tuning_factor = 0.06  # << STARTEN SIE MIT DIESEM WERT (0.005 oder 0.001)

    # kinetic constants
    k_MeOH_CO = tuning_factor * 1.88e8 * np.exp(- 113711/ (R * T)) # HIER ÄNDERN
    k_RWGS = tuning_factor * 1.16e10 * np.exp(- 126573/ (R * T))   # WGS     #[mol/kg_cat*s*bar**0.5]
    k_MeOH = tuning_factor * 7.08e4 * np.exp(- 68252/ (R * T))     # CO2     #[mol/kg_cat*s*bar**1.5]

    # kinetic constants
    #k_MeOH_CO = 1.88e8 * np.exp(- 113711/ (R * T))    # CO      #[mol/kg_cat*s*bar**1.5]
   

   # Adsorption equilibrium Constants (graaf)
    K_CO2 =1.02e-7 * np.exp( 67400 / (R * T))
    K_CO =7.99e-7 * np.exp( 58100 / (R * T))
    K_H2O =3.8e-10 * np.exp(80876 / (R * T))
    K_H2 =(K_H2O / (4.13e-11 * np.exp(104500 / (R * T))))**2

    # Adsorption equilibrium Constants (Coteron)
    #K_CO = np.exp(-68.7 / R) * np.exp(3200 / (R * T))
    #K_CO2 = np.exp(36.8 / R) * np.exp(0 / (R * T))
    #K_H2 = np.exp(3.4 / R) * np.exp(6300 / (R * T))
    #K_H2O =3.8e-10 * np.exp(80876 / (R * T))

    
    # Calculate driving forces (reaction numerators)
    EQ1 = (P_CO2 * P_H2**1.5 - ((P_H2O * P_MeOH) / (Keq_m * P_H2**1.5)))
    EQ2 = (P_CO2 * P_H2 - ((P_CO * P_H2O) / ( Keq_rwgs)))
    EQ3 = (P_CO * P_H2**1.5 - (P_MeOH / (P_H2**0.5 * Keq_m_co)))
    
    # Denominator for all reactions (LHHW formalism)
    term1_CO = (1 + K_CO * P_CO)
    term2_CO2 =  (1+ K_CO2 * P_CO2)
    denom = (1+ K_H2**0.5 * P_H2**0.5 + K_H2O *P_H2O)
    
    # Calculate rates with limits for numerical stability
    r_MeOH =k_MeOH * K_CO2 * EQ1 / (term2_CO2 * denom)
    r_WGS =-(k_RWGS * K_CO2 * EQ2) / (term2_CO2 * denom)
    r_MeOH_CO =k_MeOH_CO * K_CO * EQ3 / (term1_CO * denom)
    r_RWGS =  -r_WGS
    
    return r_MeOH, r_RWGS, r_MeOH_CO, k_MeOH_CO, k_RWGS, k_MeOH, K_CO2, K_CO, K_H2O, K_H2,Keq_m, Keq_rwgs, Keq_m_co,term1_CO, term2_CO2, denom, EQ1, EQ2, EQ3

# ======================= Reactor Model Function with Energy Balance =======================

def plug_flow_reactor(z, state, u_s_initial, T_feed, C_total_initial, T_wall):
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
    min_conc = 1e-30
    
    # Extract concentrations and temperature from state vector
    C = state[:-1]  # Concentrations [mol/m³]
    T = state[-1]   # Temperature [K]
    
    # Ensure positive concentrations
    C = np.array(C, dtype=float)
    C = np.maximum(C, min_conc)
    
    # Calculate partial pressures (in bar)
    RT = R * T
    P = np.maximum(C * RT, min_conc) / 1e5  # Convert to bar
    
    # Get reaction rates
    r_MeOH, r_RWGS, r_MeOH_CO, *_ = rate_model(
        P[0],  # P_CO2
        P[1],  # P_CO
        P[2],  # P_H2
        P[3],  # P_MeOH
        P[4],  # P_H2O
        T
    )

    # Calculate local gas velocity considering both stoichiometric and thermal effects
    C_total_local = np.sum(C)
    C_total_local = max(C_total_local, min_conc) 
    u_s_local = u_s_initial * (C_total_local / C_total_initial) * (T / T_feed)
    
    # Calculate reaction heat generation rates (J/m³·s)
    q_MeOH = rb * r_MeOH * -dH_MeOH * 1000  # Convert from kJ/mol to J/mol
    q_RWGS = rb * r_RWGS * -dH_RWGS * 1000
    q_MeOH_CO = rb * r_MeOH_CO * -dH_MeOH_CO * 1000
    
    # Total heat generation rate (J/m³·s)
    q_rxn = q_MeOH + q_RWGS + q_MeOH_CO
    
    # Heat transfer to wall (J/m³·s)
    # A/V = perimeter/area = 4/diameter for circular tube
    q_wall = (4 * U / reactor_diameter) * (T_wall - T)

    # Calculate heat capacity of mixture (J/m³·K)
    Cp_mix = (C[0] * Cp['CO2'] + C[1] * Cp['CO'] + C[2] * Cp['H2'] + 
              C[3] * Cp['MeOH'] + C[4] * Cp['H2O']) / np.sum(C)
    
    # Volumetric heat capacity (J/m³·K)
    rho_Cp = C_total_local * Cp_mix

    reaction_terms = np.array([
        rb * (-r_MeOH - r_RWGS),     # CO2
        rb * (r_RWGS -r_MeOH_CO),               # CO
        rb * (-3*r_MeOH - r_RWGS - 2*r_MeOH_CO),   # H2
        rb * (r_MeOH + r_MeOH_CO),               # MeOH
        rb * (r_MeOH + r_RWGS)       # H2O
    ])
    
    # Calculate derivatives with safety checks
    dCdz = np.clip(reaction_terms / u_s_local, -1e7, 1e7)

    
    # Energy balance:
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
        P_CO2_i * 1e5 / (R * T_feed),
        P_CO_i * 1e5 / (R * T_feed),
        P_H2_i * 1e5 / (R * T_feed),
        P_MeOH_i * 1e5 / (R * T_feed),
        P_H2O_i * 1e5 / (R * T_feed)
    ])
    
    # Initial state vector [CO2, CO, H2, MeOH, H2O, T]
    initial_state = np.append(initial_concentrations, T_feed)
    
    # Set up evaluation points
    z_eval = np.linspace(0, reactor_length, 100)
    
    C_total_initial = np.sum(initial_concentrations)

    # Create partial function for ODE solver
    partial_reactor = partial(
        plug_flow_reactor,
        u_s_initial =u_s_initial,
        T_feed = T_feed,
        T_wall = T_wall,
        C_total_initial = C_total_initial,
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
        max_step= reactor_length / 100
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
        'r_MeOH_CO': np.zeros_like(z_positions),
        'dCdz_CO2': np.zeros_like(z_positions),
        'dCdz_CO': np.zeros_like(z_positions),
        'dCdz_H2': np.zeros_like(z_positions),
        'dCdz_MeOH': np.zeros_like(z_positions),
        'dCdz_H2O': np.zeros_like(z_positions),
        'dTdz': np.zeros_like(z_positions),
        'velocity': np.zeros_like(z_positions),
        'q_rxn': np.zeros_like(z_positions),
        'q_wall': np.zeros_like(z_positions),
        'denom': np.zeros_like(z_positions),
        'EQ1': np.zeros_like(z_positions), 
        'EQ2': np.zeros_like(z_positions),
        'EQ3': np.zeros_like(z_positions)  

    }
    
    # Calculate pressures, reaction rates, and derivatives at each position
    for i, z in enumerate(z_positions):
        # Extract concentrations and temperature at this position
        C = C_profiles[:, i]
        T = T_profile[i]
        
        # Calculate partial pressures (bar)
        RT = R * T
        P = np.maximum(C * RT, 1e-20) / 1e5
        
        # Store pressures
        results['P_CO2'][i] = P[0]
        results['P_CO'][i] = P[1]
        results['P_H2'][i] = P[2]
        results['P_MeOH'][i] = P[3]
        results['P_H2O'][i] = P[4]
        
       # Calculate reaction rates
        r_MeOH, r_RWGS, r_MeOH_CO, k_MeOH_CO_i, k_RWGS_i, k_MeOH_i, K_CO2_i, K_CO_i, K_H2O_i, K_H2_i,Keq_m_i, Keq_rwgs_i, Keq_m_co_i,term1_CO_i, term2_CO2_i, denom_i, EQ1_i, EQ2_i, EQ3_i  = rate_model(
            P[0], P[1], P[2], P[3], P[4],
            T
        )
        
        # Store reaction rates
        results['r_MeOH'][i] = r_MeOH
        results['r_RWGS'][i] = r_RWGS
        results['r_MeOH_CO'][i] = r_MeOH_CO
        results['denom'][i] = denom_i
        results['EQ1'][i] = EQ1_i
        results['EQ2'][i] = EQ2_i
        results['EQ3'][i] = EQ3_i
        
        # Calculate reaction heat generation rates (J/m³·s)
        q_MeOH = rb * r_MeOH * -dH_MeOH * 1000  # Convert from kJ/mol to J/mol
        q_RWGS = rb * r_RWGS * -dH_RWGS * 1000
        q_MeOH_CO = rb * r_MeOH_CO * -dH_MeOH_CO * 1000
    
        # Total heat generation rate (J/m³·s)
        q_rxn = q_MeOH + q_RWGS + q_MeOH_CO

        
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
    initial_P_CO = results['P_CO'][0]
    results['X_CO2'] = (initial_P_CO2 - results['P_CO2']) / initial_P_CO2 * 100  # Conversion (%)
    results['Y_MeOH'] = results['P_MeOH'] / (initial_P_CO2 + initial_P_CO) * 100  # Yield (%)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_filename = f"Park1.6_{int(GHSV)}_1h_{int(T_feed)}_{int(p_total)}_bar.csv"
    df.to_csv( output_filename, sep=';', decimal='.', index=False, encoding='utf-8')
    print(f"Reactor profile saved to {output_filename}")
    
    return df

# ======================= Plotting Function =======================

def plot_reactor_profile(df, T_wall):
    """Create plots of the reactor profile"""
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
    
    # Plot 1: Partial Pressures
    
    ax1.plot(df['z'], df['P_CO'], 'r-', label='CO')
    ax1.plot(df['z'], df['P_MeOH'], 'm-', label='MeOH')
    ax1.plot(df['z'], df['P_H2O'], 'c-', label='H₂O')
    ax1.set_xlabel('Reactor Position (m)')
    ax1.set_ylabel('Partial Pressure (bar)')
    ax1.set_title('Partial Pressures Along Reactor')
    ax1.legend()
    ax1.grid(True)
    # Rechte y-Achse: H2
    ax5 = ax1.twinx()
    ax5.plot(df['z'], df['P_H2'], 'g-', label='H₂')
    ax5.plot(df['z'], df['P_CO2'], 'b-', label='CO₂')
    ax5.set_ylabel('Partial Pressure H₂, CO2 (bar)', color='g')
    ax5.tick_params(axis='y', labelcolor='g')

    # Legenden zusammenführen
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    
    # Plot 2: Reaction Rates
    ax2.plot(df['z'], df['r_MeOH'], 'b-', label='r_MeOH')
    ax2.plot(df['z'], df['r_RWGS'], 'r-', label='r_RWGS')
    ax2.plot(df['z'], df['r_MeOH_CO'], 'g-', label='r_MeOH_CO')
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

    #plt.show()


# ======================= Main Function =======================

def main():
    """Main function to run simulation and generate plots"""
    print("Starting reactor simulation with energy balance...")

    # Verschiedene GHSV-Werte Temperaturen und Drücke testen
    T_feed_list_K = [423.15, 433.15, 443.15, 453.15, 463.15, 473.15, 483.15, 493.15, 503.15, 513.15, 523.15, 533.15, 543.15, 553.15, 563.15, 573.15, 583.15, 593.15, 603.15, 613.15, 623.15]  # in Kelvin (150-350°C in 10K Schritten)
    GHSV_list = [7800]  # in 1/h (Auf einen Wert gesetzt, um T zu isolieren)
    p_total_list = [ 50]  # in Pa (50 bar)

    # Gaszusammensetzung:
    min_val = 1e-20
    y_CO2 = 0.25
    y_H2 = 0.75
    y_CO = min_val
    y_MeOH = min_val
    y_H2O = min_val
    y_H2 = y_H2 - 3*min_val

    summary_results = []

     # Nimm den ersten (und einzigen) Druckwert für die GG-Berechnung
    p_total_eq = p_total_list[0] 

    for p_total in p_total_list:

        # Berechne initiale Partialdrücke
        P_CO2_i = y_CO2 * p_total
        P_H2_i = y_H2 * p_total
        P_CO_i = y_CO * p_total
        P_MeOH_i = y_MeOH * p_total
        P_H2O_i = y_H2O * p_total

        for T_feed in T_feed_list_K:
            T_wall = T_feed
            for GHSV in GHSV_list:
                T_ref = 273.15  # K
                P_ref = 1.01325  # bar
                porosity = 0.39
                T_ref = 273.15  
                p_ref = 1.01325


                C_CO2_i = P_CO2_i * 1e5 / (R * T_feed)
                C_H2_i = P_H2_i * 1e5 / (R * T_feed)
                C_CO_i = P_CO_i * 1e5 / (R * T_feed)
                C_MeOH_i = P_MeOH_i * 1e5 / (R * T_feed)
                C_H2O_i = P_H2O_i * 1e5 / (R * T_feed)

                u_s_norm = GHSV * reactor_length * (1- porosity) * 1/3600
                u_s_initial = (u_s_norm * (P_ref / p_total) * (T_feed / T_ref)) 

                print(f"\nSimulating for GHSV = {GHSV} 1/h, T_feed = {T_feed:.1f} K")
                print(f"Gas velocity = {u_s_initial}, Feed composition: CO2={P_CO2_i} bar, H2={P_H2_i} bar,, Wall temperature: {T_wall} K")
                print(f"A_cross = {A_cross:.4f} m², V_r = {V_r:.4f} m³, m_cat = {m_cat:.4f} kg")

                r_MeOH_feed, r_RWGS_feed, r_MeOH_CO_feed, k_MeOH_CO_feed, k_RWGS_feed, k_MeOH_feed, K_CO2_feed, K_CO_feed, K_H2O_feed, K_H2_feed, Keq_m_feed, Keq_rwgs_feed, Keq_m_co_feed, term1_CO, term2_CO2, term3, df_MeOH, df_RWGS, df_MeOH_CO = rate_model(
                    P_CO2_i, P_CO_i, P_H2_i, P_MeOH_i, P_H2O_i,
                    T_feed
                    )

                # Run simulation
                df = simulate_reactor(GHSV, u_s_initial, T_feed, p_total, y_CO2, y_H2, y_CO, y_MeOH, y_H2O, T_wall)

                if df is not None:
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
                    print(f"Maximum temperature: {T_max:.1f} K at position {df['z'].iloc[df['T'].argmax()]:.3f} m")

                    total_heat_rxn = np.trapezoid(df['q_rxn'], df['z'])
                    total_heat_wall = np.trapezoid(df['q_wall'], df['z'])
                    print(f"Net heat generated by reactions: {total_heat_rxn:.2f} kW/m")
                    print(f"Net heat transferred to wall: {total_heat_wall:.2f} kW/m")
                    print(f"Heat balance closure: {100 * (1 - (total_heat_rxn + total_heat_wall) / total_heat_rxn):.2f}%")

                    summary_results.append({
                        'GHSV': GHSV,
                        'Gas _velocity': u_s_initial,
                        'T_feed': T_feed,
                        'C_CO2_feed': C_CO2_i,
                        'C_H2_feed': C_H2_i,
                        'T_exit': T_out,
                        'T_max': T_max,
                        'p_total': p_total,

                        'r_MeOH_feed': r_MeOH_feed,
                        'r_RWGS_feed': r_RWGS_feed,
                        'r_MeOH_CO_feed': r_MeOH_CO_feed,
                        'K_CO2_feed': K_CO2_feed,
                        'K_CO_feed': K_CO_feed,
                        'K_H2O_feed': K_H2O_feed,
                        'K_H2_feed': K_H2_feed,
                        'Keq_m_feed': Keq_m_feed,
                        'Keq_rwgs_feed': Keq_rwgs_feed,
                        'Keq_m_co_feed': Keq_m_co_feed,
                        'k_MeOH_CO_feed': k_MeOH_CO_feed,
                        'k_RWGS_feed': k_RWGS_feed,
                        'k_MeOH_feed': k_MeOH_feed,
                        'term1_CO_feed': term1_CO,
                        'term2_CO2_feed': term2_CO2,
                        'term3_feed': term3,
                        'df_MeOH_feed': df_MeOH,
                        'df_RWGS_feed': df_RWGS,
                        'df_MeOH_CO_feed': df_MeOH_CO,
                        'P_MeOH': df['P_MeOH'].iloc[exit_idx],
                        'CO2_conversion_%': X_CO2,
                        'MeOH_yield_%': Y_MeOH,
                        'r_MeOH_exit': df['r_MeOH'].iloc[exit_idx],
                        'r_RWGS_exit': df['r_RWGS'].iloc[exit_idx],
                        'C_MeOH_exit': df['C_MeOH'].iloc[exit_idx],
                        'CO_exit': df['C_CO'].iloc[exit_idx],
                        'C_H2O_exit': df['C_H2O'].iloc[exit_idx],
                        'C_H2_exit': df['C_H2'].iloc[exit_idx],
                        'C_CO2_exit': df['C_CO2'].iloc[exit_idx],
                        'P_MeOH_exit': df['P_MeOH'].iloc[exit_idx],
                        'P_H2_exit': df['P_H2'].iloc[exit_idx],
                        'P_CO_exit': df['P_CO'].iloc[exit_idx],
                        'P_CO2_exit': df['P_CO2'].iloc[exit_idx],
                        'P_H2O_exit': df['P_H2O'].iloc[exit_idx],
                    })

        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv("Park_summary.csv", index=False, sep=';')
        print("\nAll simulation results saved to Park_summary.csv")
        print(summary_df)

if __name__ == "__main__":
    main()  
