# model.jl - V58: "UNCHAINED"
# Node Penalty = 0.0. Edge Penalty = 0.0.
# The model has ZERO friction. It only cares about the Mask Reward.

using Gen
using Statistics
using LinearAlgebra
using StaticArrays
using PythonCall 

if !isdefined(Main, :RoadGraph); include("graph_types.jl"); end
if !isdefined(Main, :AppearanceModel); include("appearance.jl"); end

println("✅ model.jl (V58 - Zero Friction) LOADED!")

# --- TUNABLE CONSTANTS ---
const EDGE_EXISTENCE_PENALTY = 0.0   # FREE.
const EDGE_LENGTH_PENALTY = 0.01     # Tiny drag just to stop infinite spirals.
const NODE_PENALTY = 0.0             # FREE. Add as many nodes as you need to connect.

const MASK_REWARD_SCALE = 40.0       # Huge reward.

global USE_VAE_LIKELIHOOD = false
global VAE_WEIGHT = 0.7
global APPEARANCE_WEIGHT = 0.3

function compute_graph_log_probability(graph::RoadGraph, 
                                       rgb_image::Array{Float32, 3}, 
                                       app_model::Any)
    
    if isempty(graph.nodes); return -100000.0; end 
    
    total_score = 0.0
    
    # 1. PRIORS 
    num_nodes = length(graph.nodes)
    num_edges = length(graph.edges)
    total_len = 0.0
    
    for e in graph.edges
        p1 = graph.nodes[e.start_idx]
        p2 = graph.nodes[e.end_idx]
        total_len += sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
    end
    
    # Minimal Penalties
    total_score -= (num_nodes * NODE_PENALTY)
    total_score -= (num_edges * EDGE_EXISTENCE_PENALTY)
    total_score -= (total_len * EDGE_LENGTH_PENALTY)

    # 2. LIKELIHOOD
    mask_score = 0.0
    for e in graph.edges
        path_score = 0.0
        points = 0
        for pt in e.path_points
            prob = pt[3]
            if prob > 0.2
                path_score += prob * MASK_REWARD_SCALE
            else
                path_score -= 1.0 # Very low penalty for gaps. Jump over shadows!
            end
            points += 1
        end
        if points > 0; mask_score += path_score; end
    end
    total_score += mask_score

    return total_score
end

@gen function road_graph_model(initial_graph::RoadGraph,
                               soft_mask::Matrix{Float32},
                               rgb_image::Array{Float32, 3},
                               appearance_model::Any)
    total_score = compute_graph_log_probability(initial_graph, rgb_image, appearance_model)
    @trace(factor(total_score), :score)
    return initial_graph
end

export road_graph_model, compute_graph_log_probability