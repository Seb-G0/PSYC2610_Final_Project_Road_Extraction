# utilities.jl - Helper functions

using Statistics
using LinearAlgebra
using Images

function load_inputs(mask_path::String, image_path::String)
    """Load probability mask and RGB image."""
    
    # Load mask (grayscale, values 0-1)
    mask_img = load(mask_path)
    soft_mask = Float32.(gray.(mask_img))
    
    # Load RGB image
    rgb_img = load(image_path)
    rgb_array = Float32.(channelview(rgb_img))  # [3, H, W]
    rgb_array = permutedims(rgb_array, (2, 3, 1))  # [H, W, 3]
    
    return soft_mask, rgb_array
end

function sample_path_probabilities(start_node::RoadNode, 
                                  end_node::RoadNode,
                                  soft_mask::Matrix{Float32},
                                  spacing::Int=5)
    """Sample probability values along line between nodes."""
    
    x1, y1 = start_node.x, start_node.y
    x2, y2 = end_node.x, end_node.y
    
    distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    n_samples = max(2, Int(ceil(distance / spacing)))
    
    path = SVector{3, Float32}[]
    h, w = size(soft_mask)
    
    for i in 0:(n_samples-1)
        t = i / (n_samples - 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        ix = Int(round(clamp(x, 1, w)))
        iy = Int(round(clamp(y, 1, h)))
        
        prob = soft_mask[iy, ix]
        push!(path, SVector{3, Float32}(x, y, prob))
    end
    
    return path
end

function detect_keypoints_nms(mask::Matrix{Float32}; 
                              threshold::Float32=0.5f0,
                              radius::Int=16)
    """Non-maximum suppression for junction detection."""
    
    h, w = size(mask)
    keypoints = RoadNode[]
    suppressed = falses(h, w)
    
    for y in (radius+1):(h-radius)
        for x in (radius+1):(w-radius)
            if mask[y, x] > threshold && !suppressed[y, x]
                # Check local maximum
                is_max = true
                for dy in -radius:radius, dx in -radius:radius
                    if mask[y+dy, x+dx] > mask[y, x]
                        is_max = false
                        break
                    end
                end
                
                if is_max
                    push!(keypoints, RoadNode(Float32(x), Float32(y)))
                    
                    # Suppress neighborhood
                    for dy in -radius:radius, dx in -radius:radius
                        yy, xx = y + dy, x + dx
                        if 1 <= yy <= h && 1 <= xx <= w
                            suppressed[yy, xx] = true
                        end
                    end
                end
            end
        end
    end
    
    return keypoints
end

function extract_initial_graph(soft_mask::Matrix{Float32})
    """Extract initial graph from SAM-Road mask."""
    
    # Detect keypoints
    nodes = detect_keypoints_nms(soft_mask)
    
    if length(nodes) < 2
        return RoadGraph()
    end
    
    # Connect nearby nodes
    edges = RoadEdge[]
    
    for i in 1:length(nodes)
        for j in (i+1):length(nodes)
            dist = sqrt((nodes[i].x - nodes[j].x)^2 + (nodes[i].y - nodes[j].y)^2)
            
            if dist < 150  # Max connection distance
                path = sample_path_probabilities(nodes[i], nodes[j], soft_mask)
                probs = [p[3] for p in path]
                
                if mean(probs) > 0.3 && minimum(probs) > 0.15
                    edge = RoadEdge(
                        i, j, path,
                        mean(probs),
                        minimum(probs),
                        maximum(probs)
                    )
                    push!(edges, edge)
                end
            end
        end
    end
    
    return RoadGraph(nodes, edges)
end

export load_inputs, sample_path_probabilities, extract_initial_graph