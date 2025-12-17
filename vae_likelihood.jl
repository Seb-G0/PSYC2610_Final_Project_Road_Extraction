# vae_likelihood.jl
# ROBUST PYTHON BRIDGE VERSION
# Fixes: TypeError by explicit Numpy conversion

using PythonCall
using Images
using ImageFiltering

println("✅ vae_likelihood.jl LOADING (NUMPY FIX)...")

# Global Variables
global vae_wrapper = nothing
global DATA_CONFIG = nothing
global VAE_INITIALIZED = false

# =========================================================================
# 1. EMBEDDED PYTHON BRIDGE
# =========================================================================
const PYTHON_BRIDGE_CODE = """
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

class VAEWrapper:
    def __init__(self, checkpoint_path, code_dir):
        # Ensure code_dir is in path for imports
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
            
        try:
            import vae_decoder
            import config
        except ImportError as e:
            print(f"  [Python] Import Error: {e}")
            raise e

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [Python] Device: {self.device}")

        self.model_config = config.MODEL_CONFIG
        self.data_config = config.DATA_CONFIG
        self.img_size = self.data_config["image_size"]

        self.model = vae_decoder.RoadVAE(self.model_config).to(self.device)
        
        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        if self.device.type == 'cpu':
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)

        if hasattr(checkpoint, "keys") and "model_state_dict" in checkpoint.keys():
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()

    def compute_score(self, img_numpy, mask_numpy, scale_factor):
        # Ensure input is standard numpy (handle ArrayValue wrapper if passed)
        img_numpy = np.array(img_numpy)
        mask_numpy = np.array(mask_numpy)

        with torch.no_grad():
            # Prep Image
            img_t = torch.from_numpy(img_numpy).float().to(self.device)
            if len(img_t.shape) == 3: # HWC -> CHW
                 img_t = img_t.permute(2, 0, 1).unsqueeze(0)
            
            img_t = (img_t - 0.5) / 0.5
            
            # Prep Mask
            mask_t = torch.from_numpy(mask_numpy).float().to(self.device)
            if len(mask_t.shape) == 2: # HW -> 11HW
                mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            
            # VAE Forward
            recon, mu, logvar = self.model(img_t, mask_t)
            
            # Loss Calculation
            mask_3ch = mask_t.repeat(1, 3, 1, 1)
            road_mask = (mask_3ch > 0.1).float()
            road_sum = road_mask.sum().item()
            
            if road_sum > 0:
                recon_loss = F.l1_loss(recon * road_mask, img_t * road_mask, reduction='sum') / road_sum
            else:
                recon_loss = 1.0
                
            return -float(recon_loss) * scale_factor

    def reconstruct(self, img_numpy, mask_numpy):
        img_numpy = np.array(img_numpy)
        mask_numpy = np.array(mask_numpy)
        
        with torch.no_grad():
            img_t = torch.from_numpy(img_numpy).float().to(self.device)
            if len(img_t.shape) == 3: 
                 img_t = img_t.permute(2, 0, 1).unsqueeze(0)
            img_t = (img_t - 0.5) / 0.5
            
            mask_t = torch.from_numpy(mask_numpy).float().to(self.device)
            if len(mask_t.shape) == 2:
                mask_t = mask_t.unsqueeze(0).unsqueeze(0)

            recon, _, _ = self.model(img_t, mask_t)
            
            recon = (recon * 0.5) + 0.5
            recon = torch.clamp(recon, 0, 1)
            return recon.squeeze(0).permute(1, 2, 0).cpu().numpy()

def create_wrapper(ckpt, code):
    return VAEWrapper(ckpt, code)
"""

# =========================================================================
# 2. JULIA INTERFACE
# =========================================================================

function init_vae_model(checkpoint_path::String="../vae_decoder/checkpoints/best_model.pt")
    global vae_wrapper, DATA_CONFIG, VAE_INITIALIZED
    
    println("Initializing VAE model (via Python Bridge)...")
    
    current_dir = pwd()
    candidates = [
        joinpath(dirname(current_dir), "vae_decoder"), 
        joinpath(current_dir, "../vae_decoder"),       
        abspath("../vae_decoder")                      
    ]
    
    code_dir = ""
    for c in candidates
        if isdir(c) && isfile(joinpath(c, "vae_decoder.py"))
            code_dir = abspath(c)
            break
        end
    end
    
    if isempty(code_dir)
        code_dir = abspath("../vae_decoder")
    end
    
    checkpoint_abspath = abspath(checkpoint_path)
    
    # Execute Python logic
    pyexec(PYTHON_BRIDGE_CODE, Main)
    
    try
        create_fn = pyeval("create_wrapper", Main)
        vae_wrapper = create_fn(checkpoint_abspath, code_dir)
        
        DATA_CONFIG = vae_wrapper.data_config
        img_size = pyconvert(Int, DATA_CONFIG["image_size"])
        
        VAE_INITIALIZED = true
        println("✓ VAE Bridge Established! (Img Size: $img_size)")
        
    catch e
        println("❌ Failed to initialize Python Bridge")
        rethrow(e)
    end
end

function is_vae_initialized()
    return VAE_INITIALIZED
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
# 4. LIKELIHOOD FUNCTION (With Explicit Conversion)
# =========================================================================

function compute_vae_likelihood(graph::RoadGraph,
                               rgb_image::AbstractArray;
                               scale_factor::Float64=100.0)
    global vae_wrapper, VAE_INITIALIZED, DATA_CONFIG
    
    if !VAE_INITIALIZED
        error("VAE not initialized")
    end
    
    target_size = pyconvert(Int, DATA_CONFIG["image_size"])
    
    # 1. Julia Processing
    img_resized = imresize(rgb_image, (target_size, target_size))
    hypothesis_mask = graph_to_mask(graph, (target_size, target_size))
    
    # 2. EXPLICIT CONVERSION TO NUMPY
    np = pyimport("numpy")
    img_np = np.array(img_resized)
    mask_np = np.array(hypothesis_mask)
    
    # 3. Call Python Bridge
    score_py = vae_wrapper.compute_score(img_np, mask_np, scale_factor)
    
    return pyconvert(Float64, score_py)
end

function compute_hybrid_likelihood(graph::RoadGraph,
                                  rgb_image::Array{Float32, 3},
                                  appearance_model::AppearanceModel;
                                  vae_weight::Float64=0.7,
                                  appearance_weight::Float64=0.3)
    vae_ll = compute_vae_likelihood(graph, rgb_image)
    appearance_ll = 0.0
    for edge in graph.edges
        appearance_ll += edge_appearance_score(edge, rgb_image, appearance_model)
    end
    return vae_weight * vae_ll + appearance_weight * appearance_ll
end

function get_vae_reconstruction(graph::RoadGraph, rgb_image::AbstractArray)
    global vae_wrapper, DATA_CONFIG
    
    target_size = pyconvert(Int, DATA_CONFIG["image_size"])
    img_resized = imresize(rgb_image, (target_size, target_size))
    hypothesis_mask = graph_to_mask(graph, (target_size, target_size))
    
    # Explicit conversion here too
    np = pyimport("numpy")
    img_np = np.array(img_resized)
    mask_np = np.array(hypothesis_mask)
    
    recon_py = vae_wrapper.reconstruct(img_np, mask_np)
    recon_julia = pyconvert(Array{Float32}, recon_py)
    
    return recon_julia
end

println("✅ vae_likelihood.jl LOADED!")

export init_vae_model, compute_vae_likelihood, compute_hybrid_likelihood, graph_to_mask
export is_vae_initialized, get_vae_reconstruction