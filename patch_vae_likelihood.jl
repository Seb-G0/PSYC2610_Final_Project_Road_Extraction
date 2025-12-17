# patch_vae_likelihood.jl
# Patch-based VAE likelihood computation for MCMC
# Uses PythonCall (NOT PyCall) - follows same pattern as vae_likelihood.jl

using PythonCall
using Images
using ImageFiltering

println("✅ patch_vae_likelihood.jl LOADING (PythonCall)...")

# Global Variables
global patch_vae_wrapper = nothing
global PATCH_VAE_INITIALIZED = false
global PATCH_SIZE = 128
global GRID_SIZE = 8
global IMAGE_SIZE = 1024

# =========================================================================
# 1. EMBEDDED PYTHON BRIDGE FOR PATCH VAE
# =========================================================================
# In patch_vae_likelihood.jl, REPLACE the PATCH_VAE_PYTHON_CODE string with this:

const PATCH_VAE_PYTHON_CODE = """
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

class PatchVAEWrapper:
    def __init__(self, checkpoint_path, code_dir="."):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Rx: Python Bridge initializing on {self.device}")

        # 1. SETUP PATHS
        if code_dir and code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        
        # 2. IMPORT ARCHITECTURE
        try:
            import patch_vae
            import config
        except ImportError as e:
            print(f"Rx: CRITICAL IMPORT ERROR in {code_dir}: {e}")
            raise e

        # 3. INITIALIZE MODEL
        print("Rx: Building RoadPatchVAE architecture...")
        try:
            self.model = patch_vae.RoadPatchVAE(config.MODEL_CONFIG)
        except AttributeError:
            if hasattr(patch_vae, 'create_model'):
                self.model = patch_vae.create_model(config.MODEL_CONFIG)
            else:
                self.model = patch_vae.RoadPatchVAE(**config.MODEL_CONFIG)
            
        self.model.to(self.device)

        # 4. LOAD WEIGHTS
        print(f"Rx: Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint

        self.model.eval()
        
        # 5. CONFIGURATION
        self.patch_size = 128
        self.beta = 0.01  # Strict penalty from Epoch 100

    def score_nodes(self, sat_image, road_mask, nodes):
        if not nodes: return -10.0

        patches_sat = []
        patches_mask = []
        C, H, W = sat_image.shape
        half = self.patch_size // 2
        
        for (x, y) in nodes:
            cx, cy = int(x), int(y)
            x1, x2 = max(0, cx-half), min(W, cx+half)
            y1, y2 = max(0, cy-half), min(H, cy+half)
            
            sat_crop = sat_image[:, y1:y2, x1:x2]
            mask_crop = road_mask[:, y1:y2, x1:x2]
            
            pad_h, pad_w = self.patch_size-(y2-y1), self.patch_size-(x2-x1)
            if pad_h > 0 or pad_w > 0:
                sat_crop = F.pad(sat_crop, (0, pad_w, 0, pad_h))
                mask_crop = F.pad(mask_crop, (0, pad_w, 0, pad_h))
            
            patches_sat.append(sat_crop)
            patches_mask.append(mask_crop)
            
        batch_sat = torch.stack(patches_sat).to(self.device)
        batch_mask = torch.stack(patches_mask).to(self.device)
        
        with torch.no_grad():
            # Forward Pass
            recon, mu, logvar = self.model(batch_sat, batch_mask)
            
            # --- FIX: COMPARE IMAGE TO IMAGE (3ch vs 3ch) ---
            # We want to know: Did the model successfully reconstruct the road pixels?
            
            # 1. Expand mask to 3 channels for weighting
            mask_weight = batch_mask.expand_as(recon)
            
            # 2. Calculate Squared Error between Recon and Real Image
            # (Using MSE implies Gaussian Likelihood)
            diff = (recon - batch_sat) ** 2
            
            # 3. Focus strictness on the ROAD part
            # If the mask says "Road", the image MUST look like a road.
            # We weight road pixels higher to penalize hallucinations/mismatches.
            weighted_diff = diff * (1.0 + 4.0 * mask_weight)
            
            # Sum error per patch
            recon_loss = weighted_diff.view(len(nodes), -1).sum(1)
            
            # 4. KL Divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Total Negative Log Likelihood
            total_loss = recon_loss + (self.beta * kl_loss)
            
            # Return mean negative loss (Score)
            return -torch.mean(total_loss).item()

    def get_config(self):
        return {"patch_size": self.patch_size, "grid_size": 8, "image_size": 1024}

# Factory function for Julia
def create_patch_vae_wrapper(checkpoint_path, code_dir="."):
    return PatchVAEWrapper(checkpoint_path, code_dir)
"""

# =========================================================================
# 2. JULIA INTERFACE - INITIALIZATION
# =========================================================================

function init_patch_vae(checkpoint_path::String)
    global patch_vae_wrapper, PATCH_VAE_INITIALIZED, PATCH_SIZE, GRID_SIZE, IMAGE_SIZE
    
    println("Initializing Patch VAE model (via Python Bridge)...")
    
    # Find vae_decoder directory
    current_dir = pwd()
    candidates = [
        joinpath(dirname(current_dir), "vae_decoder"), 
        joinpath(current_dir, "../vae_decoder"),       
        abspath("../vae_decoder")                      
    ]
    
    code_dir = ""
    for c in candidates
        if isdir(c) && isfile(joinpath(c, "patch_vae.py"))
            code_dir = abspath(c)
            break
        end
    end
    
    if isempty(code_dir)
        code_dir = abspath("../vae_decoder")
    end
    
    checkpoint_abspath = abspath(checkpoint_path)
    
    println("  Code dir: $code_dir")
    println("  Checkpoint: $checkpoint_abspath")
    
    # Execute Python logic
    pyexec(PATCH_VAE_PYTHON_CODE, Main)
    
    try
        create_fn = pyeval("create_patch_vae_wrapper", Main)
        patch_vae_wrapper = create_fn(checkpoint_abspath, code_dir)
        
        # Get config
        config = patch_vae_wrapper.get_config()
        PATCH_SIZE = pyconvert(Int, config["patch_size"])
        GRID_SIZE = pyconvert(Int, config["grid_size"])
        IMAGE_SIZE = pyconvert(Int, config["image_size"])
        
        PATCH_VAE_INITIALIZED = true
        println("✓ VAE Bridge Established! (Img Size: $IMAGE_SIZE, Patch: $PATCH_SIZE, Grid: $GRID_SIZE)")
        
    catch e
        println("❌ Failed to initialize Python Bridge")
        rethrow(e)
    end
end

# Alias for compatibility
init_vae_model(path::String) = init_patch_vae(path)

function is_vae_initialized()
    return PATCH_VAE_INITIALIZED
end

# =========================================================================
# 3. RASTERIZATION (Pure Julia)
# =========================================================================

function graph_to_mask(graph::RoadGraph, size::Tuple{Int, Int}=(1024, 1024))
    h, w = size
    mask = zeros(Float32, h, w)
    
    for edge in graph.edges
        edge_prob = edge.mean_prob
        for i in 2:length(edge.path_points)
            p1 = edge.path_points[i-1]
            p2 = edge.path_points[i]
            x1, y1 = Int(round(p1[1])), Int(round(p1[2]))
            x2, y2 = Int(round(p2[1])), Int(round(p2[2]))
            draw_line!(mask, x1, y1, x2, y2, edge_prob, width=5)
        end
    end
    
    mask = imfilter(mask, Kernel.gaussian(1.5))
    mask = clamp.(mask, 0.0f0, 1.0f0)
    return mask
end

function draw_line!(mask::Matrix{Float32}, x1::Int, y1::Int, x2::Int, y2::Int, 
                   probability::Float32; width::Int=5)
    h, w = size(mask)
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = x1 < x2 ? 1 : -1
    sy = y1 < y2 ? 1 : -1
    err = dx - dy
    x, y = x1, y1
    
    while true
        for dy_off in -width:width
            for dx_off in -width:width
                dist = sqrt(Float32(dx_off^2 + dy_off^2))
                if dist <= width
                    xx, yy = x + dx_off, y + dy_off
                    if 1 <= xx <= w && 1 <= yy <= h
                        mask[yy, xx] = max(mask[yy, xx], probability * (1.0f0 - dist/width))
                    end
                end
            end
        end
        if x == x2 && y == y2; break; end
        e2 = 2 * err
        if e2 > -dy; err -= dy; x += sx; end
        if e2 < dx; err += dx; y += sy; end
    end
end

# =========================================================================
# 4. LIKELIHOOD FUNCTIONS
# =========================================================================

function compute_full_likelihood(rgb_image::AbstractArray, 
                                 hypothesis_mask::AbstractArray;
                                 return_patch_grid::Bool=false)
    global patch_vae_wrapper, PATCH_VAE_INITIALIZED, IMAGE_SIZE
    
    if !PATCH_VAE_INITIALIZED
        error("Patch VAE not initialized! Call init_patch_vae() first.")
    end
    
    # Resize if needed
    img_resized = imresize(rgb_image, (IMAGE_SIZE, IMAGE_SIZE))
    mask_resized = imresize(hypothesis_mask, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to numpy explicitly
    np = pyimport("numpy")
    img_np = np.array(Float32.(img_resized))
    mask_np = np.array(Float32.(mask_resized))
    
    # Call Python
    result = patch_vae_wrapper.compute_full_likelihood(img_np, mask_np)
    
    total_ll = pyconvert(Float64, result[0])
    
    if return_patch_grid
        patch_grid = pyconvert(Matrix{Float32}, result[1])
        return total_ll, patch_grid
    else
        return total_ll
    end
end

function compute_vae_likelihood(graph::RoadGraph,
                               rgb_image::AbstractArray;
                               scale_factor::Float64=10.0)
    global PATCH_VAE_INITIALIZED, IMAGE_SIZE
    
    if !PATCH_VAE_INITIALIZED
        error("Patch VAE not initialized! Call init_patch_vae() first.")
    end
    
    # Rasterize graph to mask
    h, w = size(rgb_image)[1:2]
    hypothesis_mask = graph_to_mask(graph, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Compute patch-based likelihood
    likelihood = compute_full_likelihood(rgb_image, hypothesis_mask)
    
    # Scale for MCMC
    return Float64(likelihood * scale_factor)
end

function compute_hybrid_likelihood(graph::RoadGraph,
                                  rgb_image::Array{Float32, 3},
                                  appearance_model::AppearanceModel;
                                  vae_weight::Float64=0.7,
                                  appearance_weight::Float64=0.3)
    # VAE likelihood (patch-based)
    vae_ll = compute_vae_likelihood(graph, rgb_image)
    
    # Appearance likelihood
    appearance_ll = 0.0
    for edge in graph.edges
        appearance_ll += edge_appearance_score(edge, rgb_image, appearance_model)
    end
    
    # Weighted combination
    return vae_weight * vae_ll + appearance_weight * appearance_ll
end

# =========================================================================
# 5. VISUALIZATION HELPERS
# =========================================================================

function get_vae_reconstruction(rgb_image::AbstractArray, hypothesis_mask::AbstractArray)
    global patch_vae_wrapper, PATCH_VAE_INITIALIZED, IMAGE_SIZE
    
    if !PATCH_VAE_INITIALIZED
        error("Patch VAE not initialized!")
    end
    
    # Resize if needed
    img_resized = imresize(rgb_image, (IMAGE_SIZE, IMAGE_SIZE))
    mask_resized = imresize(hypothesis_mask, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to numpy
    np = pyimport("numpy")
    img_np = np.array(Float32.(img_resized))
    mask_np = np.array(Float32.(mask_resized))
    
    # Get reconstruction
    recon_py = patch_vae_wrapper.get_reconstruction(img_np, mask_np)
    recon_julia = pyconvert(Array{Float32, 3}, recon_py)
    
    return recon_julia
end

function visualize_patch_likelihoods(rgb_image::AbstractArray, hypothesis_mask::AbstractArray)
    global GRID_SIZE, IMAGE_SIZE
    
    _, patch_grid = compute_full_likelihood(rgb_image, hypothesis_mask, return_patch_grid=true)
    
    # Create overlay
    img_resized = Float32.(imresize(rgb_image, (IMAGE_SIZE, IMAGE_SIZE)))
    overlay = copy(img_resized)
    stride = IMAGE_SIZE ÷ GRID_SIZE
    
    # Normalize for visualization
    max_ll = maximum(abs.(patch_grid))
    if max_ll > 0
        norm_grid = patch_grid ./ max_ll
    else
        norm_grid = patch_grid
    end
    
    # Color-code patches
    for row in 1:GRID_SIZE
        for col in 1:GRID_SIZE
            y_start = (row - 1) * stride + 1
            x_start = (col - 1) * stride + 1
            y_end = min(y_start + stride - 1, IMAGE_SIZE)
            x_end = min(x_start + stride - 1, IMAGE_SIZE)
            
            ll_value = norm_grid[row, col]
            
            if ll_value >= 0
                # High likelihood = green tint
                overlay[y_start:y_end, x_start:x_end, 2] .+= 0.2f0 * ll_value
            else
                # Low likelihood = red tint
                overlay[y_start:y_end, x_start:x_end, 1] .+= 0.2f0 * abs(ll_value)
            end
        end
    end
    
    return clamp.(overlay, 0.0f0, 1.0f0), patch_grid
end

# =========================================================================
# 6. CHECKPOINT COMPARISON
# =========================================================================

function compare_checkpoints(checkpoints::Dict{String, String},
                            rgb_image::AbstractArray,
                            good_mask::AbstractArray,
                            bad_mask::AbstractArray)
    
    results = Dict{String, NamedTuple{(:good_ll, :bad_ll, :discrimination), Tuple{Float64, Float64, Float64}}}()
    
    for (name, path) in checkpoints
        if !isfile(path)
            println("⚠ Checkpoint not found: $path")
            continue
        end
        
        println("\n→ Loading: $name")
        
        try
            # Re-initialize with this checkpoint
            init_patch_vae(path)
            
            # Compute likelihoods
            good_ll = compute_full_likelihood(rgb_image, good_mask)
            bad_ll = compute_full_likelihood(rgb_image, bad_mask)
            discrimination = good_ll - bad_ll
            
            results[name] = (good_ll=good_ll, bad_ll=bad_ll, discrimination=discrimination)
            
            println("    Good LL: $(round(good_ll, digits=3))")
            println("    Bad LL:  $(round(bad_ll, digits=3))")
            println("    Δ: $(round(discrimination, digits=3))")
            
        catch e
            println("    ❌ Error: $e")
        end
    end
    
    return results
end

println("✅ patch_vae_likelihood.jl LOADED!")

export init_patch_vae, init_vae_model, is_vae_initialized
export compute_vae_likelihood, compute_hybrid_likelihood, compute_full_likelihood
export graph_to_mask, get_vae_reconstruction, visualize_patch_likelihoods
export compare_checkpoints