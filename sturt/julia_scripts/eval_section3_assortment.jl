using Combinatorics
using Printf

include("utils.jl")

function parse_assortment_arg(arg::String)
    items = split(strip(arg), ",")
    out = Int[]
    for s in items
        t = strip(s)
        if isempty(t)
            continue
        end
        push!(out, parse(Int, t))
    end
    return sort(unique(out))
end

function main()
    # section_3 fixed instance
    n = 4
    past_assortments = [[0, 2, 3, 4], [0, 1, 2, 4]]
    revenues = [10, 20, 30, 100]
    r = Dict(i => revenues[i] for i = 1:n)
    r[0] = 0
    v = Dict(
        1 => Dict(0 => 0.3, 2 => 0.3, 3 => 0.3, 4 => 0.1),
        2 => Dict(0 => 0.3, 1 => 0.3, 2 => 0.1, 4 => 0.3),
    )

    # all rankings for section_3 evaluator
    Σ = []
    for perm in permutations(0:n)
        push!(Σ, Dict(i - 1 => perm[i] for i = 1:(n + 1)))
    end

    if length(ARGS) < 1
        error("Usage: julia eval_section3_assortment.jl \"0,1,2,4\"")
    end
    x_you = parse_assortment_arg(ARGS[1])
    if 0 ∉ x_you
        error("Assortment must include outside option 0. Got: $(x_you)")
    end
    if any(i < 0 || i > n for i in x_you)
        error("Assortment contains out-of-range product id. Valid range is 0..$(n). Got: $(x_you)")
    end

    ro_you, _ = EvaluateAssortment(x_you, r, past_assortments, v, n, Σ, nothing, false)

    best_ro = -Inf
    best_S = Int[]
    for S in combinations(0:n)
        # keep the same feasibility style as section_3 demo (must include 0 and n)
        if (0 ∉ S) || (n ∉ S)
            continue
        end
        S_vec = sort(collect(S))
        ro_val, _ = EvaluateAssortment(S_vec, r, past_assortments, v, n, Σ, nothing, false)
        if ro_val > best_ro
            best_ro = ro_val
            best_S = S_vec
        end
    end

    gap = (best_ro - ro_you) / best_ro

    println("============================================================")
    println("Section-3 Robust Benchmark Evaluation (sturt metric)")
    println("============================================================")
    println("x_you assortment: ", x_you)
    @printf("RO(x_you): %.6f\n", ro_you)
    println("RO-optimal assortment under same metric: ", best_S)
    @printf("RO*: %.6f\n", best_ro)
    @printf("optimality_gap: %.6f (%.2f%%)\n", gap, 100 * gap)
end

main()
