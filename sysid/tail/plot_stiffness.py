import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root
from paths import TAIL_BENCH  # noqa: E402

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

def main():
    parser = argparse.ArgumentParser(description="Plot Quasi-Static Stiffness Calibration.")
    parser.add_argument("--file", type=str, required=True, help="Path to the calibration CSV.")
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        if os.path.exists(os.path.join(str(TAIL_BENCH / 'stiffness'), file_path)):
            file_path = os.path.join(str(TAIL_BENCH / 'stiffness'), file_path)
        else:
            print(f"Error: File '{file_path}' not found.")
            return

    # Load data
    currents_up = []
    ticks_up = []
    currents_down = []
    ticks_down = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        meta_keys = next(reader)
        meta_vals = next(reader)
        pulley_radius = float(meta_vals[0])
        
        next(reader) # Header
        
        data = list(reader)
        max_idx = 0
        max_curr = -1
        
        for i, row in enumerate(data):
            c = float(row[0])
            if c > max_curr:
                max_curr = c
                max_idx = i
                
        for row in data[:max_idx+1]:
            currents_up.append(float(row[0]))
            ticks_up.append(float(row[1]))
            
        for row in data[max_idx:]:
            currents_down.append(float(row[0]))
            ticks_down.append(float(row[1]))

    def ticks_to_mm(ticks):
        revs = np.array(ticks) / 4096.0
        return revs * (2 * np.pi * pulley_radius)

    mm_up = ticks_to_mm(ticks_up)
    mm_down = ticks_to_mm(ticks_down)
    
    mm = 1 / 25.4
    fig, ax = plt.subplots(figsize=(88*mm, 70*mm))
    
    ax.plot(mm_up, currents_up, 'o-', color='tab:blue', label='Increasing Torque (Bending)')
    ax.plot(mm_down, currents_down, 's--', color='tab:orange', label='Decreasing Torque (Relaxing)')
    
    ax.plot(mm_up[-1], currents_up[-1], 'r*', markersize=8)
    ax.annotate(f"Max: {mm_up[-1]:.1f}mm\n@ {currents_up[-1]}mA", 
                (mm_up[-1], currents_up[-1]), 
                xytext=(-10, -20), textcoords='offset points', ha='right',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Tendon Pull Displacement (mm)')
    ax.set_ylabel('Motor Current / Torque (mA)')
    ax.set_title(f'Quasi-Static Stiffness Characterization\n(Pulley R={pulley_radius}mm)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    plots_dir = os.path.join("plots", "stiffness")
    os.makedirs(plots_dir, exist_ok=True)
    filename_only = os.path.basename(file_path).replace(".csv", "")
    
    fig.savefig(os.path.join(plots_dir, f"plot_{filename_only}.png"), dpi=300)
    fig.savefig(os.path.join(plots_dir, f"plot_{filename_only}.pdf"))
    
    print(f"Plot saved to {plots_dir}/plot_{filename_only}.png")

if __name__ == "__main__":
    main()
