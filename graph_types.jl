using StaticArrays

# Node with position and uncertainty
struct RoadNode
    x::Float32
    y::Float32
end

# Edge with probability profile along path
struct RoadEdge
    start_idx::Int
    end_idx::Int
    path_points::Vector{SVector{3, Float32}}  # (x, y, probability)
    mean_prob::Float32
    min_prob::Float32
    max_prob::Float32
end

# Complete road graph
struct RoadGraph
    nodes::Vector{RoadNode}
    edges::Vector{RoadEdge}
end

# Empty graph constructor
RoadGraph() = RoadGraph(RoadNode[], RoadEdge[])

# Copy constructor
Base.copy(g::RoadGraph) = RoadGraph(copy(g.nodes), copy(g.edges))

# Appearance model (learned from image)
# UPDATED: 'threshold' replaces 'exploration_boost'
struct AppearanceModel
    road_color_median::SVector{3, Float32}
    road_color_std::SVector{3, Float32}
    threshold::Float32 
end

export RoadNode, RoadEdge, RoadGraph, AppearanceModel
