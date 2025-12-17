# priors.jl - V58: "Minimalist"
# Triangle Penalty: 10.0 (Negligible).
# Angle Constraints: Weak.

using Gen, LinearAlgebra, Statistics
if !isdefined(Main, :RoadGraph); include("graph_types.jl"); end
println("✅ priors.jl (V58 - Minimal) LOADED!")

function get_vector(n1, n2); return (n2.x - n1.x, n2.y - n1.y); end

function get_angle_between(v1, v2)
    dot_prod = v1[1]*v2[1] + v1[2]*v2[2]
    m1 = sqrt(v1[1]^2 + v1[2]^2); m2 = sqrt(v2[1]^2 + v2[2]^2)
    if m1 == 0 || m2 == 0; return 0.0; end
    val = clamp(dot_prod / (m1 * m2), -1.0, 1.0)
    return rad2deg(acos(val))
end

function detect_triangles(graph::RoadGraph)
    adj = [Int[] for _ in 1:length(graph.nodes)]
    for e in graph.edges; push!(adj[e.start_idx], e.end_idx); push!(adj[e.end_idx], e.start_idx); end
    triangles = 0
    for i in 1:length(graph.nodes)
        for n1 in adj[i]
            if n1 <= i; continue; end
            for n2 in adj[n1]
                if n2 <= n1 || n2 == i; continue; end
                if i in adj[n2]; triangles += 1; end
            end
        end
    end
    return triangles
end

function compute_log_prior(graph::RoadGraph)
    penalty = 0.0
    
    # 1. Triangles: Almost ignored.
    n_tri = detect_triangles(graph)
    penalty += 10.0 * n_tri 
    
    adj = [Int[] for _ in 1:length(graph.nodes)]
    for e in graph.edges; push!(adj[e.start_idx], e.end_idx); push!(adj[e.end_idx], e.start_idx); end
    
    for i in 1:length(graph.nodes)
        neighbors = adj[i]; deg = length(neighbors)
        if deg < 2; continue; end
        n_curr = graph.nodes[i]
        
        # Only punish extreme zig-zags (< 90 degrees)
        if deg == 2
            n1 = graph.nodes[neighbors[1]]; n2 = graph.nodes[neighbors[2]]
            v1 = get_vector(n_curr, n1); v2 = get_vector(n_curr, n2)
            angle = get_angle_between(v1, v2)
            if angle < 90.0
                penalty += (90.0 - angle) * 1.0 
            end
        end
    end
    
    # Sparsity (Weak)
    penalty += length(graph.edges) * 1.0
    
    return -penalty
end

export compute_log_prior