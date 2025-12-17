using Gen
using Statistics
using StaticArrays
using LinearAlgebra
using Images

if !isdefined(Main, :RoadGraph); include("graph_types.jl"); end

println("✅ appearance.jl (V36 - Expansionist) LOADED!")

function learn_appearance_model(graph::RoadGraph, rgb_image::Array{Float32, 3})
    h, w, c = size(rgb_image)
    road_pixels = SVector{3, Float32}[]
    
    # 1. Collect pixels
    for edge in graph.edges
        for point in edge.path_points
            x, y = Int(round(point[1])), Int(round(point[2]))
            if 1 <= x <= w && 1 <= y <= h
                push!(road_pixels, SVector{3, Float32}(rgb_image[y, x, 1], rgb_image[y, x, 2], rgb_image[y, x, 3]))
            end
        end
    end
    
    # 2. Fallback
    if isempty(road_pixels)
        cx, cy = Int(round(w/2)), Int(round(h/2))
        for dy in -2:2, dx in -2:2
             if 1 <= cx+dx <= w && 1 <= cy+dy <= h
                 push!(road_pixels, SVector{3, Float32}(rgb_image[cy+dy, cx+dx, 1], rgb_image[cy+dy, cx+dx, 2], rgb_image[cy+dy, cx+dx, 3]))
             end
        end
    end
    
    # 3. Last Resort
    if isempty(road_pixels)
        return AppearanceModel(SVector{3, Float32}(0.4, 0.4, 0.4), SVector{3, Float32}(0.2, 0.2, 0.2), 0.2)
    end
    
    # 4. Component-wise Median
    r_vals = [p[1] for p in road_pixels]
    g_vals = [p[2] for p in road_pixels]
    b_vals = [p[3] for p in road_pixels]
    
    med = SVector{3, Float32}(median(r_vals), median(g_vals), median(b_vals))
    
    # 5. Robust Deviation
    devs = [norm(p - med) for p in road_pixels]
    mad = median(devs)
    
    # Threshold: 3.0x MAD
    thresh = min(0.35f0, max(0.12f0, Float32(mad * 3.0)))
    
    println("🎨 Learned Road Color: $med | MAD: $mad | Threshold: $thresh")
    
    return AppearanceModel(med, SVector{3, Float32}(mad, mad, mad), thresh)
end

function is_tree_occluded(rgb::SVector{3, Float32})
    r, g, b = rgb[1], rgb[2], rgb[3]
    is_green = (g > r + 0.03) && (g > b + 0.03)
    is_dark = (r + g + b) / 3.0 < 0.15
    return is_green || is_dark
end

function is_safe_color(rgb::SVector{3, Float32}, model::AppearanceModel)
    r, g, b = rgb[1], rgb[2], rgb[3]
    if (g > r + 0.05) && (g > b + 0.05); return false; end 
    d = norm(rgb - model.road_color_median)
    # Allow exploration up to 1.5x
    return d < (model.threshold * 1.5)
end

function edge_appearance_score(edge::RoadEdge,
                               rgb_image::Array{Float32, 3},
                               model::AppearanceModel)
    h, w, c = size(rgb_image)
    total_score = 0.0
    
    # The "Optimistic" limit.
    # We give points for anything closer than 1.6x the threshold.
    limit = model.threshold * 1.6
    
    for point in edge.path_points
        x, y = Int(round(point[1])), Int(round(point[2]))
        if x < 1 || x > w || y < 1 || y > h; continue; end
        
        rgb = SVector{3, Float32}(rgb_image[y, x, 1], rgb_image[y, x, 2], rgb_image[y, x, 3])
        dist = norm(rgb - model.road_color_median)
        
        if is_tree_occluded(rgb)
            # Neutral/Slight positive for occlusion (doubt benefits the defendant)
            total_score += 0.5 
        elseif dist < limit
            # REWARD: Even "okay" matches get points.
            # Max score 10.0 for perfect match.
            # Score decreases linearly to 0.0 at the limit.
            score = (1.0 - (dist / limit)) * 10.0
            total_score += score
        else
            # PENALTY: Only applies if strictly outside the generous limit
            penalty = (dist - limit) * 5.0
            total_score -= min(penalty, 2.0) # Cap penalty lightly
        end
    end
    
    return total_score
end

export learn_appearance_model, edge_appearance_score, is_safe_color, AppearanceModel
