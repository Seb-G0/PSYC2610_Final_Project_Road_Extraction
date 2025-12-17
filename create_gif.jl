# create_gif.jl - Create GIF from saved frames
# Updated to use PythonCall to prevent kernel crashes

using FileIO
using Images
using PythonCall

"""
Create animated GIF from frame directory using pure Julia.
Note: Requires ImageMagick installed on the system for 'save' to work with GIFs.
"""
function create_gif_from_frames(frame_dir::String,
                                output_path::String;
                                fps::Int=10,
                                final_frame_duration::Int=3)
    
    println("\n[Creating GIF - Pure Julia]")
    println("  Reading frames from: $frame_dir")
    
    # Get all frame files
    frame_files = sort(filter(f -> startswith(basename(f), "frame_") && endswith(f, ".png"),
                             readdir(frame_dir, join=true)))
    
    if isempty(frame_files)
        error("No frames found in $frame_dir")
    end
    
    println("  Found $(length(frame_files)) frames")
    
    # Load frames
    frames = []
    for (i, frame_file) in enumerate(frame_files)
        if i % 10 == 0
            println("    Loading frame $i/$(length(frame_files))...")
        end
        img = load(frame_file)
        push!(frames, img)
    end
    
    # Duplicate final frame for longer display
    final_frame = frames[end]
    n_final_duplicates = fps * final_frame_duration
    for _ in 1:n_final_duplicates
        push!(frames, final_frame)
    end
    
    println("  Total frames (with final hold): $(length(frames))")
    println("  Saving GIF to: $output_path")
    
    # Save as GIF
    try
        save(output_path, cat(frames..., dims=3), fps=fps)
        println("✓ GIF created successfully!\n")
    catch e
        println("⚠ Pure Julia GIF creation failed (likely missing ImageMagick).")
        println("  Error: $e")
        println("  Trying Python method instead...")
        return create_gif_python(frame_dir, output_path, fps=fps, final_frame_duration=final_frame_duration)
    end
    
    return output_path
end

"""
Create GIF using Python's imageio (Robust).
Uses PythonCall to avoid conflicts with other parts of the pipeline.
"""
function create_gif_python(frame_dir::String,
                          output_path::String;
                          fps::Int=10,
                          final_frame_duration::Int=3)
        
    # Import imageio via PythonCall
    try
        imageio = pyimport("imageio")
    catch e
        error("Python 'imageio' not found. Please pip install imageio.")
    end
    
    println("\n[Creating GIF with Python imageio]")
    
    # Get frame files
    frame_files = sort(filter(f -> startswith(basename(f), "frame_") && endswith(f, ".png"),
                             readdir(frame_dir, join=true)))
    
    if isempty(frame_files)
        println("⚠ No frames found in $frame_dir")
        return
    end

    println("  Found $(length(frame_files)) frames")
    
    # Use a Python List to store images to avoid conversion overhead/errors
    py_images = pylist()
    
    for frame_file in frame_files
        # Let Python read the file directly
        img = imageio.imread(frame_file)
        py_images.append(img)
    end
    
    # Add final frame repeats
    if length(py_images) > 0
        final_img = py_images[end] # Get last element
        for _ in 1:(fps * final_frame_duration)
            py_images.append(final_img)
        end
    end
    
    println("  Saving GIF to $output_path...")
    
    # Save
    imageio.mimsave(output_path, py_images, fps=fps)
    
    println("✓ GIF created: $output_path\n")
    
    return output_path
end

export create_gif_from_frames, create_gif_python