import os
import glob
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root
from paths import TAIL_BENCH  # noqa: E402

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

def main():
    parser = argparse.ArgumentParser(description="Analyze Dynamic Benchmark Data")
    parser.add_argument("--env", type=str, required=True, choices=["air", "water"], help="Environment to analyze")
    parser.add_argument("--dataset", type=str, default="default", help="Dataset name to filter files")
    parser.add_argument("--max_rpm", type=float, default=61.0, help="Max theoretical RPM for plotting reference")
    args = parser.parse_args()

    # Load all relevant files matching dataset and env
    search_pattern = os.path.join(str(TAIL_BENCH / args.env), f"*_{args.dataset}_Benchmark_{args.env}_*.csv")
    files = glob.glob(search_pattern)
    
    # Also support older format for backward compatibility if dataset wasn't used
    if not files and args.dataset == "default":
        search_pattern = os.path.join(str(TAIL_BENCH / args.env), f"*_Benchmark_{args.env}_*.csv")
        files = glob.glob(search_pattern)
        
    if not files:
        print(f"No benchmark data found for dataset '{args.dataset}' and environment '{args.env}'")
        return
        
    results = []
    for f in files:
        with open(f, 'r') as file:
            reader = csv.reader(file)
            keys = next(reader)
            vals = next(reader)
            
            # Handle new format vs old format
            if 'Dataset' in keys:
                env_val = vals[0]
                dataset_val = vals[1]
                pulley_radius = float(vals[2])
                cmd_freq = float(vals[3])
                frac_amp = float(vals[4])
                travel_amp = float(vals[5])
                ticks_amp = float(vals[6])
            else:
                env_val = vals[0]
                cmd_freq = float(vals[1])
                frac_amp = float(vals[2])
                ticks_amp = float(vals[3])
                pulley_radius = 12.5 # default
                travel_amp = (ticks_amp / 4096.0) * (2 * np.pi * pulley_radius)
            
            header = next(reader)
            times, actual_pos, goal_pos, vel, load = [], [], [], [], []
            for row in reader:
                if not row: continue
                times.append(float(row[0]))
                actual_pos.append(float(row[1]))
                goal_pos.append(float(row[2]))
                
                # Check if Velocity is in the row (old versions might not have it or have it at different indices)
                # Currently we have: Time (s), Position, GoalPosition, Velocity, Load
                if len(row) > 4:
                    vel.append(float(row[3]))
                    load.append(float(row[4]))
                else:
                    vel.append(0.0)
                    load.append(float(row[3]) if len(row) == 4 else 0.0) # Handle arbitrary old lengths
                
        if len(actual_pos) == 0: continue
        
        actual_pos = np.array(actual_pos)
        vel = np.array(vel)
        load = np.array(load)
        
        meas_amp_ticks = (np.max(actual_pos) - np.min(actual_pos)) / 2.0
        meas_amp_travel = (meas_amp_ticks / 4096.0) * (2 * np.pi * pulley_radius)
        max_load = np.max(np.abs(load)) if len(load) > 0 else 0
        
        # Velocity in XL430 is 0.229 rpm per unit
        vel_rpm = vel * 0.229
        max_vel_rpm = np.max(np.abs(vel_rpm)) if len(vel_rpm) > 0 else 0
        
        goal_centered = np.array(goal_pos) - np.mean(goal_pos)
        actual_centered = actual_pos - np.mean(actual_pos)
        correlation = np.correlate(actual_centered, goal_centered, mode='full')
        lags = np.arange(-len(actual_pos) + 1, len(actual_pos))
        best_lag_idx = np.argmax(correlation)
        best_lag = lags[best_lag_idx]
        
        dt = times[1] - times[0] if len(times)>1 else 0.01
        time_lag = best_lag * dt
        phase_lag_deg = (time_lag * cmd_freq) * 360.0
        
        results.append({
            'Freq': cmd_freq,
            'CmdAmpFrac': frac_amp,
            'CmdAmpTravel': travel_amp,
            'MeasAmpTravel': meas_amp_travel,
            'TrackingRatio': (meas_amp_travel / travel_amp) * 100 if travel_amp > 0 else 0,
            'MaxLoad': max_load,
            'MaxVelRPM': max_vel_rpm,
            'PhaseLag': abs(phase_lag_deg)
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by=['Freq', 'CmdAmpFrac'])
    
    mm = 1 / 25.4
    fig, axs = plt.subplots(3, 2, figsize=(180*mm, 200*mm))
    
    freqs = sorted(df['Freq'].unique())
    amps = sorted(df['CmdAmpFrac'].unique())
    
    # 1. Bode Plot (Amplitude Tracking vs Frequency)
    ax = axs[0,0]
    for amp in amps:
        sub = df[df['CmdAmpFrac'] == amp]
        ax.plot(sub['Freq'], sub['TrackingRatio'], marker='o', label=f"{amp*100:.0f}\\% Amp")
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude Tracking Ratio (\\%)')
    ax.set_title(f'Bode Plot: Amplitude Response ({args.env.upper()})')
    ax.legend(title="Cmd Amp", loc='lower left', fontsize=7, ncol=2)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Phase Lag vs Frequency
    ax = axs[0,1]
    for amp in amps:
        sub = df[df['CmdAmpFrac'] == amp]
        ax.plot(sub['Freq'], sub['PhaseLag'], marker='s', label=f"{amp*100:.0f}\\% Amp")
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase Lag (Degrees)')
    ax.set_title(f'Bode Plot: Phase Delay ({args.env.upper()})')
    ax.legend(title="Cmd Amp", loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 3. Torque Saturation Heatmap
    ax = axs[1,0]
    pivot_load = df.pivot(index='Freq', columns='CmdAmpFrac', values='MaxLoad')
    if not pivot_load.empty:
        cax = ax.matshow(pivot_load.values, cmap='Reds', aspect='auto', vmin=0, vmax=1000)
        fig.colorbar(cax, ax=ax, label='Peak Torque (mA)')
        ax.set_xticks(range(len(pivot_load.columns)))
        ax.set_yticks(range(len(pivot_load.index)))
        ax.set_xticklabels([f"{x*100:.0f}\\%" for x in pivot_load.columns])
        ax.set_yticklabels(pivot_load.index)
        ax.set_xlabel('Commanded Amplitude Fraction')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Torque Demand ({args.env.upper()})')

    # 4. Absolute Measured Stroke (Travel)
    ax = axs[1,1]
    for freq in freqs:
        sub = df[df['Freq'] == freq]
        ax.plot(sub['CmdAmpFrac'] * 100, sub['MeasAmpTravel'], marker='^', label=f"{freq} Hz")
    
    max_travel = df['CmdAmpTravel'].max()
    ax.plot([0, 100], [0, max_travel], 'k--', alpha=0.5, label='Ideal')
    ax.set_xlabel('Commanded Amplitude (\\%)')
    ax.set_ylabel('Measured Amplitude Travel (mm)')
    ax.set_title(f'Absolute Kinematic Output ({args.env.upper()})')
    ax.legend(title="Frequency", loc='upper left', fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.6)

    # 5. Speed Saturation Plot
    ax = axs[2,0]
    for amp in amps:
        sub = df[df['CmdAmpFrac'] == amp]
        ax.plot(sub['Freq'], sub['MaxVelRPM'], marker='d', label=f"{amp*100:.0f}\\% Amp")
    ax.axhline(args.max_rpm, color='r', linestyle='--', label=f'Max Speed ({args.max_rpm} RPM)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak Velocity (RPM)')
    ax.set_title(f'Speed Demand ({args.env.upper()})')
    ax.legend(title="Cmd Amp", loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 6. Blank/Extra
    axs[2,1].axis('off')
    
    plt.tight_layout()
    plots_dir = os.path.join("plots", "dynamic")
    os.makedirs(plots_dir, exist_ok=True)
    
    filename = f"{args.dataset}_Dynamic_Benchmark_{args.env.upper()}"
    fig.savefig(os.path.join(plots_dir, f"{filename}.png"), dpi=300)
    fig.savefig(os.path.join(plots_dir, f"{filename}.pdf"))
    
    print(f"\nAnalysis complete! Dashboard saved to: plots/dynamic/{filename}.png")

if __name__ == "__main__":
    main()
