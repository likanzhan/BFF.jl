module BFF

using Makie

export τ², BF₁₀, bff, BFMakie

### 0. Define the BayesFactor Struct
struct BayesFactor{T <: AbstractFloat}
    BFs::Vector{T}
    taus::Vector{T}
    omegas::AbstractVector{T}
    MaxVal::T
    MaxInd::Int64
    MinVal::T
    MinInd::Int64
    tauMax::T
    omegaMax::T
end

### 1. Define τ² Function #####################################
const ωs = 0:0.001:1

τ²(::Val{:oneSample}, ω::Float64, n::Int64)             = n * ω^2 / 2
τ²(::Val{:twoSample}, ω::Float64, n₁::Int64, n₂::Int64) = n₁ * n₂ * ω^2 / (n₁ + n₂)
τ²(::Val{:count},     ω::Float64, n::Int64)             = n * ω^2
τ²(::Val{:LRT},       ω::Float64, n::Int64)             = n * ω^2
τ²(::Val{:linear},    ω::Float64, n::Int64)             = n * ω^2 / 2
τ²(format, ω::AbstractVector{Float64}, ns...)           = map(ω -> τ²(Val(format), ω, ns...), ω)

### 2. Define BF10 #####################################
function BF₁₀(::Val{:𝑧}, τ²::Float64, 𝑧)
    𝑧² = 𝑧^2
    term1 = (τ² + 1)^(-3/2)
    term2 = (1 + τ² * 𝑧² / (τ² + 1))
    term3 = exp(τ² * 𝑧² / (2 * (τ² + 1)))
    return term1 * term2 * term3
end

function BF₁₀(::Val{:𝑡}, τ²::Float64, 𝑡, 𝑑𝑓)
    𝑡² = 𝑡^2
    𝑟 = (1 + 𝑡² / 𝑑𝑓)
    𝑠 = 1 + 𝑡² / (𝑑𝑓 * (1 + τ²))
    𝑞 = τ² * (𝑑𝑓 + 1) / (𝑑𝑓 * (1 + τ²))
    term1 = (τ² + 1)^(-3/2)
    term2 = (𝑟 / 𝑠)^((𝑑𝑓 + 1)/2)
    term3 = (1 + 𝑞 * 𝑡² / 𝑠)
    return term1 * term2 * term3
end

function BF₁₀(::Val{:𝜒}, τ²::Float64, χ, 𝑑𝑓)
    term1 = (τ² + 1)^(-𝑑𝑓 / 2 - 1)
    term2 = 1 + τ² * χ / (𝑑𝑓 * (τ² + 1))
    term3 = exp(τ² * χ / (2 * (τ² + 1)))
    return term1 * term2 * term3
end

function BF₁₀(::Val{:𝐹}, τ²::Float64, 𝐹, 𝑑𝑓₁, 𝑑𝑓₂)
    v = 𝑑𝑓₂ * (τ² + 1)
    term1 = (τ² + 1)^(- 𝑑𝑓₁/2 - 1)
    term2 = ((1 + 𝑑𝑓₁ * 𝐹 / 𝑑𝑓₂) / (1 + 𝑑𝑓₁ * 𝐹 / v))^((𝑑𝑓₁ + 𝑑𝑓₂)/2)
    term3 = 1 + (𝑑𝑓₁ + 𝑑𝑓₂) * τ² * 𝐹 / (v * (1 + 𝑑𝑓₁ * 𝐹 / v))
    return term1 * term2 * term3
end

BF₁₀(test, τ²::AbstractVector{Float64}, args...)  = 
    map(τ² -> BF₁₀(Val(test), τ², args...), τ²)

### 3. Wrap-up Bayes Factor Function #####################################
function bff((format, ns...), (test, args...))
    τ²s = τ²(format, ωs, ns...)
    BFs = BF₁₀(test, τ²s, args...)
    MaxVal, MaxInd = findmax(BFs)
    MinVal, MinInd = findmin(BFs)
    τMax  = τ²s[MaxInd]
    ωMax  = ωs[MaxInd]
    BayesFactor(BFs, τ²s, ωs, MaxVal, MaxInd, MinVal, MinInd, τMax, ωMax)
end

function bff(DTs::Vector)
    length(DTs) != 2 && error("Dimension Mismatch")
    Res = Vector(undef, length(DTs))
    for (idx, value) in enumerate(DTs)
        Res[idx] = bff(value...)
    end  
    τs   = Res[1].taus.* Res[2].taus
    BFst =  Res[1].BFs .* Res[2].BFs
    MaxVal, MaxInd = findmax(BFst)
    MinVal, MinInd = findmin(BFst)
    τMax  = τs[MaxInd]
    ωMax  = ωs[MaxInd]

    BayesFactor(BFst, τs, ωs, MaxVal, MaxInd, MinVal, MinInd, τMax, ωMax)
end

### 4. Define the plot function
function BFMakie(BF::BayesFactor; xlimits = (0, 1), ylimits = (BF.MinVal, BF.MaxVal), label = nothing)
    fig = Figure()
    dt  = [x * 10^y for y in 0:3 for x in [1, 2, 5]]
    pos = [dt; 1 ./ dt[2:end]]
    lab = [string.(dt, ":1"); string.("1:", dt[2:end])]
    ax = Axis(fig[1, 1], 
        limits = (xlimits, (ylimits[1], ylimits[2] * 1.2)),
        title = "BFF",
        xticks = 0:0.2:1,
        yscale = log,
        yticks = (pos, lab), 
        xlabel = "Standardized Effect Size ω",
        ylabel = "Bayes Factor Agains Null Hypothesis (F₁₀)"
        )
    resp = [0, 0.1, 0.35, 0.65, 1]
    for (st, nd, cor) in zip(resp[1:end-1], diff(resp), [:red, :orange, :blue, :green])
        poly!(ax, Rect(st, exp(log(ylimits[1])), nd, exp(log(ylimits[2] * 1.1))), color = (cor, 0.1))
    end
        lines!(ax,  BF.omegas, BF.BFs, linewidth = 5, color = :black, label = label)
        vlines!(ax, BF.omegaMax, linestyle = :dash, color = :black)
        hlines!(ax, 1, linestyle = :solid, color = :black)
    fig
end

### 5. Replication of the examples defined in the paper
BFz = bff((:oneSample, 100), (:𝑧, 2))
BFMakie(BFz)

BFc = bff((:count, 707), (:𝜒, 12.65, 6))
BFMakie(BFc, xlimits = (0, 0.2), ylimits = (0.0025, 3))

BFf1 = let
    df1, df2 = 2, 82
    n = df1 + df2 + 1
    bff((:linear, n), (:𝐹, 4.05, df1, df2))
end

BFf2 = let
    df1, df2 = 2, 137
    n = df1 + df2 + 1
    bff((:linear, n), (:𝐹, 1.99, df1, df2))
end

BFft = let
    df11, df12 = 2, 82
    n1 = df11 + df12 + 1
    df21, df22 = 2, 137
    n2 = df21 + df22 + 1
    f_stat1, f_stat2 = 4.05, 1.99
    args = [((:linear, n1), (:𝐹, f_stat1, df11, df12)), ((:linear, n2), (:𝐹, f_stat2, df21, df22))]
    bff(args)
end
    
fig = BFMakie(BFft, ylimits = (0.005, 7), label = "Cmobined")
lines!(fig.content[1], BFf1.omegas, BFf1.BFs, linewidth = 5, linestyle = :dot,   color = :red, label = "Original")
lines!(fig.content[1], BFf2.omegas, BFf2.BFs, linewidth = 5, linestyle = :dash,  color = :green, label = "Replication")
axislegend(fig.content[1])

fig
end # Module
