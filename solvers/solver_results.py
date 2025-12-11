from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time  # <--- Added for timing
from stokes_shishken import stokes_shishken 

# === Configuration ===
tot = 6  # Number of different N values to test
base_N = 8  # Base N value to scale
eps_val = 0.01  # Epsilon value
numRefines = 1  # Number of refinements(Barycentric Splits or Incenter Splits) 

# === Initialize Arrays ===
# BC = Barycenter, IC = Incenter
unormL2_bc = np.zeros(tot)
unormH1_bc = np.zeros(tot)
pnorm_bc   = np.zeros(tot)
inf_sup_bc = np.zeros(tot)
aspect_bc  = np.zeros(tot)
time_bc    = np.zeros(tot) # <--- To hold timing data

unormL2_ic = np.zeros(tot)
unormH1_ic = np.zeros(tot)
pnorm_ic   = np.zeros(tot)
inf_sup_ic = np.zeros(tot)
aspect_ic  = np.zeros(tot)
time_ic    = np.zeros(tot) # <--- To hold timing data

# === Main Loop ===
print(f"Starting simulation with eps={eps_val}...")
total_start_time = time.perf_counter()

for i in range(tot):
    N = base_N * (i + 1)
    print(f"\n=========================================")
    print(f" Iteration {i+1}/{tot} (N={N})")
    print(f"=========================================")
    
    # --- Run Barycentric (BC) ---
    print(f"  > Running Barycentric Refinement...")
    t_start = time.perf_counter() # Start Timer
    
    # CALL SOLVER
    uL2, uH1, pL2, asp, beta = stokes_shishken(N, eps_val, numRefines, 'bc')
    
    t_end = time.perf_counter()   # End Timer
    duration = t_end - t_start
    print(f"    [Done] Time taken: {duration:.4f} seconds")
    
    # Store results
    unormL2_bc[i] = uL2
    unormH1_bc[i] = uH1
    pnorm_bc[i]   = pL2
    aspect_bc[i]  = asp
    inf_sup_bc[i] = beta
    time_bc[i]    = duration

    # --- Run Incenter (IC) ---
    print(f"  > Running Incenter Refinement")
    t_start = time.perf_counter() # Start Timer
    
    # CALL SOLVER
    uL2, uH1, pL2, asp, beta = stokes_shishken(N, eps_val, numRefines, 'ic')
    
    t_end = time.perf_counter()   # End Timer
    duration = t_end - t_start
    print(f"    [Done] Time taken: {duration:.4f} seconds")

    # Store results
    unormL2_ic[i] = uL2
    unormH1_ic[i] = uH1
    pnorm_ic[i]   = pL2
    aspect_ic[i]  = asp
    inf_sup_ic[i] = beta
    time_ic[i]    = duration

total_end_time = time.perf_counter()
print(f"\nTotal Simulation Time: {(total_end_time - total_start_time):.2f} seconds")

# === X-Axis for Plots ===
xax = np.arange(base_N, base_N * (tot + 1), base_N) 

# === Plotting (Log-Log for Convergence) ===
# Velocity L2
fig1, ax1 = plt.subplots()
ax1.loglog(xax, unormL2_bc, 'o-', label='Barycenter')
ax1.loglog(xax, unormL2_ic, 'ro-', label='Incenter')
ax1.loglog(xax, xax**(-3.0) * unormL2_bc[0] * (xax[0]**3.0), 'k--', label='Slope -3')
ax1.set_xlabel("N (1/h)")
ax1.set_ylabel("L2 error velocity")
ax1.legend()
ax1.grid(True, which="both", ls="-")
fig1.savefig("my_results/shisken_L2_velocity.png")
plt.show()

# Velocity H1
fig2, ax2 = plt.subplots()
ax2.loglog(xax, unormH1_bc, 'o-', label='Barycenter')
ax2.loglog(xax, unormH1_ic, 'ro-', label='Incenter')
ax2.loglog(xax, xax**(-2.0) * unormH1_bc[0] * (xax[0]**2.0), 'k--', label='Slope -2')
ax2.set_xlabel("N (1/h)")
ax2.set_ylabel("H1 error velocity")
ax2.legend()
ax2.grid(True, which="both", ls="-")
fig2.savefig("my_results/shisken_H1_velocity.png")
plt.show()

# Pressure L2
fig3, ax3 = plt.subplots()
ax3.loglog(xax, pnorm_bc, 'o-', label='Barycenter')
ax3.loglog(xax, pnorm_ic, 'ro-', label='Incenter')
ax3.set_xlabel("N (1/h)")
ax3.set_ylabel("L2 error pressure")
ax3.legend()
ax3.grid(True, which="both", ls="-")
fig3.savefig("my_results/shisken_L2_pressure.png")
plt.show()

# Time Comparison Plot (Linear Scale)
fig4, ax4 = plt.subplots()
ax4.plot(xax, time_bc, 'o-', label='Barycenter Time')
ax4.plot(xax, time_ic, 'ro-', label='Incenter Time')
ax4.set_xlabel("N (Resolution)")
ax4.set_ylabel("Time (seconds)")
ax4.legend()
ax4.grid(True)
fig4.savefig("my_results/execution_time_comparison.png")
plt.show()


# === Saving Data to Text Files ===

# 1. Save Velocity L2 Errors
header_txt = "N, L2_Vel_BC, L2_Vel_IC"
data_block = np.column_stack((xax, unormL2_bc, unormL2_ic))
np.savetxt("my_results/error_velocity_L2.txt", data_block, header=header_txt, comments='')

# 2. Save Pressure L2 Errors
header_txt = "N, L2_Press_BC, L2_Press_IC"
data_block = np.column_stack((xax, pnorm_bc, pnorm_ic))
np.savetxt("my_results/error_pressure_L2.txt", data_block, header=header_txt, comments='')

# 3. Save Stability Data
header_txt = "N, Aspect_BC, InfSup_BC, Aspect_IC, InfSup_IC"
data_block = np.column_stack((xax, aspect_bc, inf_sup_bc, aspect_ic, inf_sup_ic))
np.savetxt("my_results/stability_data.txt", data_block, header=header_txt, comments='')

# 4. Save Execution Times
header_txt = "N, Time_BC_Seconds, Time_IC_Seconds"
data_block = np.column_stack((xax, time_bc, time_ic))
np.savetxt("my_results/execution_times.txt", data_block, header=header_txt, comments='')

print("All results and timing data saved to /my_results/ directory.")