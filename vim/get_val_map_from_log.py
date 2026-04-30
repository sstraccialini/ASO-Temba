import argparse
import os
import logging
from datetime import datetime

def setup_logging(save_dir):
    # Create eval_logs directory if it doesn't exist
    log_dir = os.path.join(save_dir, 'eval_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'eval_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_max_val_map(log_file, logger):
    max_val_map = 0.0
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'Full-val-map: ' in line:
                    val_map = float(line.split('Full-val-map: ')[1].strip())
                    max_val_map = max(max_val_map, val_map)
        
        logger.info(f"File: {log_file}")
        logger.info(f"Max Val Map: {max_val_map:.4f}")
        return max_val_map
    
    except FileNotFoundError:
        logger.error(f"Error: File not found - {log_file}")
        return None
    except Exception as e:
        logger.error(f"Error processing file {log_file}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Get maximum validation MAP from training logs')
    parser.add_argument('--log_path', type=str, required=True, 
                      help='Path to training.log file or directory containing training logs')
    args = parser.parse_args()
    
    # Setup logging
    save_dir = os.path.dirname(args.log_path) if os.path.isfile(args.log_path) else args.log_path
    logger = setup_logging(save_dir)
    
    if os.path.isfile(args.log_path):
        # Process single file
        get_max_val_map(args.log_path, logger)
    else:
        # Process all log files in directory
        for root, _, files in os.walk(args.log_path):
            for file in files:
                if file == 'training.log':
                    log_file = os.path.join(root, file)
                    get_max_val_map(log_file, logger)

if __name__ == "__main__":
    main()