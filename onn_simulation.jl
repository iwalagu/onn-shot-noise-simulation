using LinearAlgebra
using Distributions
using Plots
using Random

Random.seed!(42)

N = 4

# ============================================================
# DEFINE 4 CLASSES BY THEIR MEAN INPUT VECTORS
# Each class has a distinct pattern that U maps to a 
# distinct output — this is what a trained ONN does
# ============================================================

# Create 4 clearly separated class prototypes
class_prototypes = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]

# Build a unitary that maps each prototype to a distinct output port
# We do this by making U = the matrix whose columns ARE the prototypes
# Since prototypes are already orthonormal, this IS a unitary
U = zeros(ComplexF64, N, N)
for i in 1:N
    U[:, i] = class_prototypes[i]
end

println("Unitarity check (should be ~0): ", norm(U'*U - I))

# Check intensity patterns
println("\nIntensity patterns per class:")
for c in 1:N
    x = ComplexF64.(class_prototypes[c])
    intensities = abs2.(U * x)
    println("  Class $c: ", round.(intensities, digits=3))
end

# ============================================================
# CLASSIFICATION WITH SHOT NOISE
# ============================================================
function one_trial(U, n_bar, class_prototypes; N=4, input_noise=0.3)
    true_class = rand(1:N)
    
    # Add small Gaussian noise to input to simulate realistic conditions
    x = ComplexF64.(class_prototypes[true_class]) .+ 
        input_noise .* randn(ComplexF64, N)
    
    # normalize
    x = x ./ norm(x)

    y = U * x
    intensities = abs2.(y)

    counts = [rand(Poisson(max(n_bar * intensities[i], 1e-10))) for i in 1:N]

    if sum(counts) == 0
        return rand(1:N) == true_class ? 1 : 0
    end

    return argmax(counts) == true_class ? 1 : 0
end

# ============================================================
# SWEEP
# ============================================================
n_trials = 2000
n_bar_values = 10 .^ range(-1, 4, length=60)
accuracies = Float64[]

println("\nRunning simulation...")
for n_bar in n_bar_values
    correct = sum(one_trial(U, n_bar, class_prototypes) for _ in 1:n_trials)
    push!(accuracies, correct / n_trials)
end
println("Done.")

# ============================================================
# SPOT CHECK
# ============================================================
println("\nSpot check:")
for i in [1, 10, 20, 30, 40, 50, 60]
    println("  n_bar = ", round(n_bar_values[i], digits=3),
            "  accuracy = ", round(accuracies[i], digits=3))
end

# ============================================================
# PLOT
# ============================================================
hbar_omega_zJ = 128.0
energies_zJ = n_bar_values .* hbar_omega_zJ

p = plot(energies_zJ, accuracies,
    xscale = :log10,
    xlabel = "Energy per MAC (zJ)",
    ylabel = "Classification Accuracy",
    title  = "ONN Accuracy vs. Energy — Shot Noise Limited (4×4 MZI Mesh)",
    label  = "Simulated accuracy",
    linewidth = 2,
    color  = :steelblue,
    marker = :circle,
    markersize = 3,
    legend = :topleft,
    grid   = true,
    gridalpha = 0.3,
    size   = (800, 500)
)

hline!([0.25], linestyle=:dash, color=:gray,
    label="Chance (25%)", linewidth=1.5)
hline!([0.90], linestyle=:dot, color=:green,
    label="90% threshold", linewidth=1.5)
vline!([50.0], linestyle=:dot, color=:red,
    label="SQL floor 50 zJ (Hamerly 2019)", linewidth=1.5)
vline!([0.66 * hbar_omega_zJ], linestyle=:dash, color=:orange,
    label="Wang et al. 2022 (0.66 photons)", linewidth=1.5)

savefig(p, "accuracy_vs_energy.png")
println("Plot saved.")

# ============================================================
# FIND KNEE
# ============================================================
knee_index = findfirst(x -> x >= 0.80, accuracies)
if knee_index !== nothing
    println("\nKnee (80% threshold):")
    println("  n_bar* = ", round(n_bar_values[knee_index], digits=2))
    println("  E*     = ", round(n_bar_values[knee_index] * 128.0, digits=1), " zJ/MAC")
end

thresh90_index = findfirst(x -> x >= 0.90, accuracies)
if thresh90_index !== nothing
    println("90% threshold:")
    println("  n_bar  = ", round(n_bar_values[thresh90_index], digits=2))
    println("  Energy = ", round(n_bar_values[thresh90_index] * 128.0, digits=1), " zJ/MAC")
end
p2 = plot(n_bar_values, accuracies,
    xscale = :log10,
    xlabel = "Mean photon number n̄",
    ylabel = "Classification Accuracy",
    title  = "ONN Accuracy vs. Photon Number (4×4 MZI Mesh)",
    label  = "Simulated accuracy",
    linewidth = 2,
    color  = :steelblue,
    marker = :circle,
    markersize = 3,
    legend = :topleft,
    grid = true,
    gridalpha = 0.3,
    size = (800, 500)
)

hline!([0.25], linestyle=:dash, color=:gray,
    label="Chance (25%)", linewidth=1.5)
hline!([0.90], linestyle=:dot, color=:green,
    label="90% threshold", linewidth=1.5)
vline!([0.66], linestyle=:dash, color=:orange,
    label="Wang et al. 2022 (0.66 photons)", linewidth=1.5)
vline!([2.76], linestyle=:dot, color=:red,
    label="Simulation knee n̄★ ≈ 2.76", linewidth=1.5)

savefig(p2, "accuracy_vs_nbar.png")
println("n̄ plot saved.")