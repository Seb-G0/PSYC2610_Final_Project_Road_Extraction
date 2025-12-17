using Gen, Random, Statistics, LinearAlgebra, StaticArrays
if !isdefined(Main, :RoadGraph); include("graph_types.jl"); end
println("✅ proposals.jl (V58 - Force Connect) LOADED!")

mutable struct ProposalContext
    rgb_image::Array{Float32, 3}
    app_model::Any
end
const PROPOSAL_CTX = ProposalContext(zeros(Float32, 1,1,1), nothing)

function set_proposal_context(rgb, model)
    PROPOSAL_CTX.rgb_image = rgb; PROPOSAL_CTX.app_model = model
end

# --- HELPERS ---
function orientation(p, q, r)
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if abs(val) < 1e-4; return 0; end
    return (val > 0) ? 1 : 2
end

function check_crossing_strict(graph::RoadGraph, p1::RoadNode, p2::RoadNode)
    for e in graph.edges
        q1 = graph.nodes[e.start_idx]; q2 = graph.nodes[e.end_idx]
        if (q1 == p1 || q1 == p2 || q2 == p1 || q2 == p2); continue; end
        o1 = orientation(p1, p2, q1); o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1); o4 = orientation(q1, q2, p2)
        if (o1 != o2 && o3 != o4); return true; end
    end
    return false
end

function get_degrees(graph)
    deg = zeros(Int, length(graph.nodes))
    for e in graph.edges; deg[e.start_idx]+=1; deg[e.end_idx]+=1; end
    return deg
end

# --- PROPOSALS ---

@gen function extend_road_proposal(trace, graph::RoadGraph, soft_mask::Matrix{Float32})
    n_nodes = length(graph.nodes); if n_nodes == 0; return (graph, false); end
    deg = get_degrees(graph)
    tips = [i for i in 1:n_nodes if deg[i] == 1]
    if isempty(tips); return (graph, false); end
    
    src_idx = tips[{:idx} ~ uniform_discrete(1, length(tips))]
    src_node = graph.nodes[src_idx]
    
    # Simple Angle Logic
    angle = {:angle} ~ uniform(0.0, 2*pi)
    dist = {:dist} ~ uniform(30.0, 70.0)
    
    px = src_node.x + cos(angle)*dist; py = src_node.y + sin(angle)*dist
    h,w = size(soft_mask)
    if px<2 || px>w-2 || py<2 || py>h-2; return (graph, false); end
    
    new_node = RoadNode(Float32(px), Float32(py))
    if check_crossing_strict(graph, src_node, new_node); return (graph, false); end
    
    path = [SVector{3,Float32}(src_node.x, src_node.y, 1f0), SVector{3,Float32}(new_node.x, new_node.y, 1f0)]
    return (RoadGraph([graph.nodes; new_node], [graph.edges; RoadEdge(src_idx, n_nodes+1, path, 0.5f0,0f0,0f0)]), true)
end

@gen function snap_tip_proposal(trace, graph::RoadGraph, soft_mask::Matrix{Float32})
    deg = get_degrees(graph)
    tips = [i for i in 1:length(graph.nodes) if deg[i] == 1]
    if isempty(tips); return (graph, false); end
    
    src_idx = tips[{:t1} ~ uniform_discrete(1, length(tips))]
    src_node = graph.nodes[src_idx]
    
    best_target = -1; min_dist = 200.0
    
    for i in 1:length(graph.nodes)
        if i == src_idx; continue; end
        n_tgt = graph.nodes[i]
        d = sqrt((src_node.x - n_tgt.x)^2 + (src_node.y - n_tgt.y)^2)
        
        if d < min_dist && d > 1.0
            # ONLY CONSTRAINT: No Crossing.
            if !check_crossing_strict(graph, src_node, n_tgt)
                min_dist = d
                best_target = i
            end
        end
    end
    
    if best_target != -1
        path = [SVector{3,Float32}(src_node.x, src_node.y, 1f0), SVector{3,Float32}(graph.nodes[best_target].x, graph.nodes[best_target].y, 1f0)]
        new_edge = RoadEdge(src_idx, best_target, path, 0.8f0, 0f0, 0f0)
        return (RoadGraph(graph.nodes, [graph.edges; new_edge]), true)
    end
    return (graph, false)
end

@gen function delete_edge_proposal(trace, graph::RoadGraph, soft_mask::Matrix{Float32})
    if isempty(graph.edges); return (graph, false); end
    idx = {:e} ~ uniform_discrete(1, length(graph.edges))
    return (RoadGraph(graph.nodes, [graph.edges[k] for k in 1:length(graph.edges) if k!=idx]), true)
end

export extend_road_proposal, snap_tip_proposal, delete_edge_proposal, set_proposal_context
