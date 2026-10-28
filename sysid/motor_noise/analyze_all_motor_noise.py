import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import glob
import os

def analyze_all():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tail/data'))
    csv_files = glob.glob(os.path.join(data_dir, '*Benchmark_air*.csv'))
    
    if not csv_files:
        print("No benchmark files found.")
        return

    print(f"Found {len(csv_files)} benchmark files. Analyzing...")

    results = []

    for f in csv_files:
        try:
            # Read metadata
            meta_df = pd.read_csv(f, nrows=1)
            if 'Freq' not in meta_df.columns or 'AmpFrac' not in meta_df.columns:
                continue
                
            freq = float(meta_df['Freq'].iloc[0])
            amp_frac = float(meta_df['AmpFrac'].iloc[0])
            
            # Read data
            df = pd.read_csv(f, skiprows=2)
            if 'Time (s)' not in df.columns or 'Velocity' not in df.columns or 'Position' not in df.columns:
                continue
                
            time = df['Time (s)'].values
            raw_vel = df['Velocity'].values
            raw_pos = df['Position'].values
            
            if len(time) < 50: # Skip very short or failed runs
                continue

            # 1. POSITION NOISE
            window_length = min(15, len(raw_pos) - 1)
            if window_length % 2 == 0: window_length -= 1
            smooth_pos = savgol_filter(raw_pos, window_length, 3)
            pos_residuals = raw_pos - smooth_pos
            pos_var_ticks = np.var(pos_residuals)
            
            # 2. VELOCITY NOISE
            vel_window = min(31, len(raw_vel) - 1)
            if vel_window % 2 == 0: vel_window -= 1
            smooth_vel = savgol_filter(raw_vel, vel_window, 3)
            vel_residuals = raw_vel - smooth_vel
            vel_var_ticks = np.var(vel_residuals)

            # Conversions
            tick_to_rad = (2 * np.pi) / 4096
            pos_var_rads = pos_var_ticks * (tick_to_rad ** 2)
            
            rpm_to_radsec = 0.229 * (2 * np.pi / 60)
            vel_var_radsec = vel_var_ticks * (rpm_to_radsec ** 2)
            
            results.append({
                'File': os.path.basename(f),
                'Frequency': freq,
                'Amplitude': amp_frac,
                'Pos_Var_rad2': pos_var_rads,
                'Vel_Var_rads2': vel_var_radsec,
                'Pos_Var_ticks2': pos_var_ticks,
                'Vel_Var_ticks2': vel_var_ticks
            })
            
        except Exception as e:
            print(f"Error processing {os.path.basename(f)}: {e}")

    if not results:
        print("No valid data processed.")
        return

    res_df = pd.DataFrame(results)
    
    # Sort and group
    res_df = res_df.sort_values(by=['Frequency', 'Amplitude'])
    
    print("\n" + "="*60)
    print("📊 AGGREGATE NOISE ANALYSIS RESULTS")
    print("="*60)
    # Print a nice pivot table for velocity variance
    print("\nVelocity Variance (rad/s)^2 vs Frequency & Amplitude:")
    pivot_vel = res_df.pivot_table(values='Vel_Var_rads2', index='Frequency', columns='Amplitude', aggfunc='mean')
    print(pivot_vel.round(4).to_string())

    print("\nPosition Variance (rad)^2 vs Frequency & Amplitude:")
    pivot_pos = res_df.pivot_table(values='Pos_Var_rad2', index='Frequency', columns='Amplitude', aggfunc='mean')
    print(pivot_pos.round(6).to_string())

    # Plotting
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot Velocity Variance
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for amp in sorted(res_df['Amplitude'].unique()):
        subset = res_df[res_df['Amplitude'] == amp].drop(columns=['File']).groupby('Frequency').mean().reset_index()
        ax1.plot(subset['Frequency'], subset['Vel_Var_rads2'], marker='o', label=f'Amp: {amp*100:.0f}%')
        
    ax1.set_title('Motor Velocity Sensor Noise (Variance) vs Movement Frequency')
    ax1.set_xlabel('Commanded Sine Wave Frequency (Hz)')
    ax1.set_ylabel('Velocity Variance ($rad/s$)$^2$')
    ax1.legend(title='Amplitude Fraction')
    ax1.grid(True)
    fig1.savefig(os.path.join(plot_dir, "aggregate_velocity_variance.png"), dpi=200)
    plt.close(fig1)

    # Plot Position Variance
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for amp in sorted(res_df['Amplitude'].unique()):
        subset = res_df[res_df['Amplitude'] == amp].drop(columns=['File']).groupby('Frequency').mean().reset_index()
        ax2.plot(subset['Frequency'], subset['Pos_Var_rad2'], marker='o', label=f'Amp: {amp*100:.0f}%')
        
    ax2.set_title('Motor Position Sensor Noise (Variance) vs Movement Frequency')
    ax2.set_xlabel('Commanded Sine Wave Frequency (Hz)')
    ax2.set_ylabel('Position Variance ($rad$)$^2$')
    ax2.legend(title='Amplitude Fraction')
    ax2.grid(True)
    ax2.set_yscale('log') # Log scale is good here because values are tiny
    fig2.savefig(os.path.join(plot_dir, "aggregate_position_variance.png"), dpi=200)
    plt.close(fig2)

    print(f"\nPlots saved to: {plot_dir}/")
    print(" - aggregate_velocity_variance.png")
    print(" - aggregate_position_variance.png")

if __name__ == '__main__':
    analyze_all()