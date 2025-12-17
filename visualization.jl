# visualization.jl - Create GIF visualizations of MCMC

using Images
using FileIO
using ImageDraw
using ColorTypes

include("graph_types.jl")

function draw_graph_on_image(rgb_image::Array{Float32, 3},
                            graph::RoadGraph;
                            node_color=RGB(1.0, 0.0, 0.0),      # Red nodes
                            edge_color=RGB(0.0, 1.0, 0.0),      # Green edges
                            node_radius::Int=8,
                            edge_width::Int=3,
                            show_probabilities::Bool=true)
    """
    Draw graph overlay on RGB satellite image.
    
    Returns new image with graph drawn on top.
    """
    
    # Convert to RGB image type
    img = copy(rgb_image)
    img_rgb = colorview(RGB, permutedims(img, (3, 1, 2)))
    img_draw = RGB.(img_rgb)  # Convert to mutable RGB array
    
    # Draw edges first (so nodes appear on top)
    for edge in graph.edges
        # Draw edge path
        for i in 2:length(edge.path_points)
            p1 = edge.path_points[i-1]
            p2 = edge.path_points[i]
            
            x1, y1 = Int(round(p1[1])), Int(round(p1[2]))
            x2, y2 = Int(round(p2[1])), Int(round(p2[2]))
            
            # Color by probability if requested
            if show_probabilities
                prob = (p1[3] + p2[3]) / 2
                # Low prob = yellow, high prob = green
                edge_col = RGB(1.0 - prob, prob, 0.0)
            else
                edge_col = edge_color
            end
            
            # Draw thick line
            draw_line!(img_draw, x1, y1, x2, y2, edge_col, edge_width)
        end
    end
    
    # Draw nodes on top
    for node in graph.nodes
        x, y = Int(round(node.x)), Int(round(node.y))
        draw_circle!(img_draw, x, y, node_radius, node_color)
    end
    
    return img_draw
end

function draw_line!(img::Matrix{RGB{Float64}}, 
                   x1::Int, y1::Int, x2::Int, y2::Int,
                   color::RGB, width::Int)
    """Draw thick line using Bresenham's algorithm."""
    
    h, w = size(img)
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = x1 < x2 ? 1 : -1
    sy = y1 < y2 ? 1 : -1
    err = dx - dy
    
    x, y = x1, y1
    
    while true
        # Draw thick point
        for dy_offset in -width:width
            for dx_offset in -width:width
                if sqrt(dx_offset^2 + dy_offset^2) <= width
                    xx = x + dx_offset
                    yy = y + dy_offset
                    if 1 <= xx <= w && 1 <= yy <= h
                        img[yy, xx] = color
                    end
                end
            end
        end
        
        if x == x2 && y == y2
            break
        end
        
        e2 = 2 * err
        if e2 > -dy
            err -= dy
            x += sx
        end
        if e2 < dx
            err += dx
            y += sy
        end
    end
end

function draw_circle!(img::Matrix{RGB{Float64}},
                     cx::Int, cy::Int, radius::Int,
                     color::RGB)
    """Draw filled circle."""
    
    h, w = size(img)
    
    for dy in -radius:radius
        for dx in -radius:radius
            if dx^2 + dy^2 <= radius^2
                x = cx + dx
                y = cy + dy
                if 1 <= x <= w && 1 <= y <= h
                    img[y, x] = color
                end
            end
        end
    end
end

function add_text_overlay!(img::Matrix{RGB{Float64}},
                          text::String,
                          x::Int, y::Int;
                          fontsize::Int=20,
                          color::RGB=RGB(1.0, 1.0, 1.0))
    """Add text to image (simple version - just draws a box with text info)."""
    
    # Draw background box
    box_width = length(text) * 10
    box_height = 30
    
    h, w = size(img)
    
    for dy in 0:box_height
        for dx in 0:box_width
            xx = x + dx
            yy = y + dy
            if 1 <= xx <= w && 1 <= yy <= h
                # Semi-transparent black background
                img[yy, xx] = RGB(0.0, 0.0, 0.0)
            end
        end
    end
    
    # Note: Actual text rendering requires more complex code
    # For now, we just have the info box
end

function create_frame(rgb_image::Array{Float32, 3},
                     graph::RoadGraph,
                     iteration::Int,
                     score::Float64,
                     acceptance_rate::Float64)
    """
    Create a single frame showing:
    - Satellite image
    - Current graph overlay
    - Iteration info
    - Score
    - Acceptance rate
    """
    
    # Draw graph on image
    frame = draw_graph_on_image(
        rgb_image, graph,
        show_probabilities=true
    )
    
    # Add info text (approximate position)
    # You could use Plots.jl or similar for better text rendering
    
    return frame
end

export draw_graph_on_image, create_frame