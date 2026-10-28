import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root
from paths import TAIL_BENCH  # noqa: E402

# LaTeX-style plotting configuration
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "savefig.dpi": 300
})

def load_benchmark_csv(filepath):
    df_meta = pd.read_csv(filepath, nrows=1)
    df_data = pd.read_csv(filepath, header=2)
    return df_meta, df_data

def process_benchmark_data(df_data):
    max_rpm = np.max(np.abs(df_data['Velocity'])) * 0.229
    max_load = np.max(np.abs(df_data['Load']))
    error_deg = (df_data['GoalPosition'] - df_data['Position']) * (360.0 / 4096.0)
    rmse_deg = np.sqrt(np.mean(error_deg**2))
    max_error_deg = np.max(np.abs(error_deg))
    return {"max_rpm": max_rpm, "max_load": max_load, "rmse_deg": rmse_deg, "max_error_deg": max_error_deg}

def find_best_files(base_dir, dist, freq, env):
    pattern = os.path.join(base_dir, env, f"*_r17.5_d{dist:04.1f}_Benchmark_{env}_{freq:03.1f}Hz_*.csv")
    files = glob.glob(pattern)
    
    amp_map = {}
    for f in files:
        try:
            amp_str = os.path.basename(f).split('_')[-1].replace("Amp.csv", "")
            amp = float(amp_str)
            # Take the latest run for each amplitude
            if amp not in amp_map or os.path.basename(f) > os.path.basename(amp_map[amp]):
                amp_map[amp] = f
        except:
            continue
    return amp_map

def generate_combined_summary(dist, frequencies):
    base_dir = str(TAIL_BENCH)   # air/ and water/ subfolders; water/ lives in the archive only
    plot_dir = "sysid/tail/plots/comparison_air_water/"
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = {'air': 'tab:blue', 'water': 'tab:red'}
    markers = {1.0: 'o', 2.0: 's', 3.0: '^'}
    linestyles = {1.0: '-', 2.0: '--', 3.0: ':'}
    
    radius = None
    
    for freq in frequencies:
        data_100pct = {}
        for env in ["air", "water"]:
            amp_map = find_best_files(base_dir, dist, freq, env)
            if not amp_map: continue
            
            res_list = []
            for amp in sorted(amp_map.keys()):
                filepath = amp_map[amp]
                df_meta, df_data = load_benchmark_csv(filepath)
                stats = process_benchmark_data(df_data)
                stats["amp"] = amp
                res_list.append(stats)
                
                if not radius: radius = df_meta['PulleyRadius_mm'][0]
                
                if amp == 100.0:
                    data_100pct[env] = df_data
                    # Still generate individual tracking plots if they don't exist
                    # (Simplified here to keep the code clean)
            
            if not res_list: continue
            df = pd.DataFrame(res_list)
            
            lbl = f"{env.capitalize()} {freq}Hz"
            fmt = f"{markers.get(freq, 'x')}{linestyles.get(freq, '-')}"
            
            axes[0].plot(df["amp"], df["max_rpm"], fmt, color=colors[env], label=lbl, markersize=4)
            axes[1].plot(df["amp"], df["max_load"], fmt, color=colors[env], label=lbl, markersize=4)
            axes[2].plot(df["amp"], df["max_error_deg"], fmt, color=colors[env], label=lbl, markersize=4)

        # Generate the combined tracking plot for this frequency if both envs exist
        if "air" in data_100pct and "water" in data_100pct:
            f_track, ax_track = plt.subplots(figsize=(6, 4))
            df_a, df_w = data_100pct["air"], data_100pct["water"]
            t, m = df_a['Time (s)'], (df_a['Time (s)'] >= 2.0) & (df_a['Time (s)'] <= 5.0)
            goal = (df_a['GoalPosition'] - df_a['GoalPosition'][0]) * (360.0/4096.0)
            act_a = (df_a['Position'] - df_a['GoalPosition'][0]) * (360.0/4096.0)
            act_w = (df_w['Position'] - df_w['GoalPosition'][0]) * (360.0/4096.0)
            ax_track.plot(t[m], goal[m], 'k--', label='Goal', alpha=0.5)
            ax_track.plot(t[m], act_a[m], label='Air', color=colors['air'])
            ax_track.plot(t[m], act_w[m], label='Water', color=colors['water'])
            ax_track.set_title(f"Tracking ($r$={radius}, $d$={dist}, $f$={freq}Hz)")
            ax_track.set_xlabel("Time (s)"), ax_track.set_ylabel("Pos (deg)"), ax_track.legend(loc='lower left'), ax_track.grid(True, alpha=0.3)
            tn = f"pos_tracking_comparison_r{radius}_d{dist}_{freq}Hz_100Amp"
            f_track.savefig(os.path.join(plot_dir, f"{tn}.png")), f_track.savefig(os.path.join(plot_dir, f"{tn}.pdf"))
            plt.close(f_track)

    axes[0].set_title("Max Speed (RPM)"), axes[1].set_title("Max Current (Units)"), axes[2].set_title("Max Error (deg)")
    for ax in axes: ax.set_xlabel("Amplitude (%)"), ax.legend(), ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Benchmark Summary ($r$={radius}mm, $d$={dist}mm)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    sn = f"summary_stats_r{radius}_d{dist}_combined_freq"
    fig.savefig(os.path.join(plot_dir, f"{sn}.png")), fig.savefig(os.path.join(plot_dir, f"{sn}.pdf"))
    plt.close(fig)
    print(f"Generated combined summary for d={dist}")

if __name__ == "__main__":
    # Generate combined summaries for d=9.0 and d=5.0
    generate_combined_summary(9.0, [1.0])
    generate_combined_summary(5.0, [1.0, 2.0])
