import os
import subprocess
import time
from pathlib import Path
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command, description):
    """Run commands"""
    logging.info(f"Starting: {description}")
    logging.info(f"Running command: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logging.info(output.strip())
        
        # Check for errors
        _, stderr = process.communicate()
        if process.returncode != 0:
            logging.error(f"Error in {description}: {stderr}")
            raise subprocess.CalledProcessError(process.returncode, command)
        
        logging.info(f"Completed: {description}")
        
    except Exception as e:
        logging.error(f"Failed to run {description}: {str(e)}")
        raise

def quote_path(path):
    """Safe check for spaces."""
    return f'"{path}"' if ' ' in str(path) else str(path)

def main():
    # Define paths
    base_dir = Path("images")
    video_dir = base_dir / "video"
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    detection_output_dir = Path("detection_outputs")
    
    # Ensure directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract frames from video
    video_files = list(video_dir.glob("*.mp4"))  # Assuming MP4 format
    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    
    video_path = video_files[0]  # Use the first video found
    frame_extractor_cmd = f"python core/frame_extractor.py {quote_path(video_path)} {quote_path(input_dir)}"
    run_command(frame_extractor_cmd, "Frame extraction")
    
    # Step 2: Run COLMAP conversion
    convert_cmd = f"python core/convert.py -s {quote_path(base_dir)}"
    run_command(convert_cmd, "COLMAP conversion")
    
    # Step 3: Train the model
    train_cmd = f"python train.py -s {quote_path(base_dir)} -m {quote_path(output_dir)} -w"
    run_command(train_cmd, "Model training")
    
    # Step 4: Check for completion
    ply_file = output_dir / "point_cloud.ply"
    if not ply_file.exists():
        raise FileNotFoundError(f"Expected point cloud file not found at {ply_file}")
    
    # Step 5: Run viewpoint detection
    logging.info("Starting viewpoint detection...")
    viewpoint_cmd = f"python core/viewpoint_detection.py"
    run_command(viewpoint_cmd, "Viewpoint detection")
    
    # Step 6: Move detection results to output folder
    detection_results = Path("detection_results.jpg")
    if detection_results.exists():
        shutil.move(str(detection_results), str(detection_output_dir / "detection_results.jpg"))
        logging.info(f"Moved detection results to {detection_output_dir}")
    else:
        logging.warning("No detection results found to move")
    
    logging.info("Pipeline completed successfully!")
    logging.info(f"Point cloud file is ready at: {ply_file}")
    logging.info(f"Detection results are saved in: {detection_output_dir}")

if __name__ == "__main__":
    main() 