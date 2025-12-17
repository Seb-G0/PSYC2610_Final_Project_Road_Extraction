using Gen, Printf, Images, LinearAlgebra, StaticArrays

if !isdefined(Main, :RoadGraph); include("graph_types.jl"); end
if !isdefined(Main, :road_graph_model); include("model.jl"); end 
if !isdefined(Main, :set_proposal_context); include("proposals.jl"); end
println("✅ inference.jl (V57) LOADED!")

function collapse_short_edges(graph::RoadGraph, threshold::Float64=15.0)
    nodes = graph.nodes; edges = graph.edges; if isempty(edges); return graph; end
    merge_map = Dict{Int, Int}()
    for e in edges
        p1 = nodes[e.start_idx]; p2 = nodes[e.end_idx]
        if sqrt((p1.x-p2.x)^2 + (p1.y-p2.y)^2) < threshold
            u, v = min(e.start_idx, e.end_idx), max(e.start_idx, e.end_idx)
            while haskey(merge_map, u); u = merge_map[u]; end
            while haskey(merge_map, v); v = merge_map[v]; end
            if u != v; merge_map[v] = u; end
        end
    end
    if isempty(merge_map); return graph; end
    new_nodes = RoadNode[]; old_to_new = Dict{Int, Int}(); c = 1
    for i in 1:length(nodes)
        if !haskey(merge_map, i); push!(new_nodes, nodes[i]); old_to_new[i] = c; c+=1; end
    end
    new_edges = RoadEdge[]; seen = Set{Tuple{Int, Int}}()
    for e in edges
        u = e.start_idx; v = e.end_idx
        while haskey(merge_map, u); u = merge_map[u]; end
        while haskey(merge_map, v); v = merge_map[v]; end
        if u != v
            u_n, v_n = old_to_new[u], old_to_new[v]
            pair = (min(u_n, v_n), max(u_n, v_n))
            if !(pair in seen)
                push!(seen, pair); push!(new_edges, RoadEdge(u_n, v_n, e.path_points, e.mean_prob,0f0,0f0))
            end
        end
    end
    return RoadGraph(new_nodes, new_edges)
end

function save_mcmc_frame(graph::RoadGraph, rgb::Array{Float32, 3}, filename::String)
    h, w, _ = size(rgb); img = copy(rgb)
    for e in graph.edges
        p1 = graph.nodes[e.start_idx]; p2 = graph.nodes[e.end_idx]
        steps = Int(round(sqrt((p1.x-p2.x)^2 + (p1.y-p2.y)^2)))
        for k in 0:steps
            t = k/max(1, steps); x = Int(round(p1.x+t*(p2.x-p1.x))); y = Int(round(p1.y+t*(p2.y-p1.y)))
            if 1<=x<=w && 1<=y<=h; img[y,x,1]=1.0; img[y,x,2]=0.0; img[y,x,3]=0.0; end
        end
    end
    try; save(filename, colorview(RGB, permutedims(img, (3, 1, 2)))); catch; end
end

function mcmc_inference(initial_graph::RoadGraph, soft_mask::Matrix{Float32}, rgb_image::Array{Float32, 3}, iterations::Int; app_model=nothing, save_frames::Bool=false, frame_dir::String="", kwargs...) 
    Main.set_proposal_context(rgb_image, app_model)
    current_graph = collapse_short_edges(initial_graph, 2.0) # Safe mild cleanup
    current_score = Main.compute_graph_log_probability(current_graph, rgb_image, app_model)
    best_graph = deepcopy(current_graph); best_score = current_score; acc_count = 0
    
    moves = [(extend_road_proposal, 0.60), (snap_tip_proposal, 0.35), (delete_edge_proposal, 0.05)]
    total_w = sum(w for (m,w) in moves); probs = cumsum([w/total_w for (m,w) in moves])
    
    println("🚀 Starting MCMC (V57 Balanced) - $iterations iters")
    if save_frames && !isempty(frame_dir); mkpath(frame_dir); save_mcmc_frame(current_graph, rgb_image, joinpath(frame_dir, "frame_000000.png")); end

    for iter in 1:iterations
        if iter % 100 == 0; current_graph = collapse_short_edges(current_graph, 5.0); end
        r = rand(); idx = findfirst(p -> r <= p, probs); prop_func = moves[idx][1]
        (prop_graph, valid) = prop_func(nothing, current_graph, soft_mask)
        if valid
            prop_score = Main.compute_graph_log_probability(prop_graph, rgb_image, app_model)
            if log(rand()) < (prop_score - current_score)
                current_graph = prop_graph; current_score = prop_score; acc_count += 1
                if current_score > best_score; best_score = current_score; best_graph = deepcopy(current_graph); end
            end
        end
        if iter % 100 == 0; print("\rIter $iter | Score: $(round(current_score, digits=1)) | Nodes: $(length(current_graph.nodes))   "); end
        if save_frames && (iter % 100 == 0); save_mcmc_frame(current_graph, rgb_image, joinpath(frame_dir, "frame_$iter.png")); end
    end
    println("\n✅ Done.")
    return best_graph, best_score, acc_count
end
export mcmc_inference
