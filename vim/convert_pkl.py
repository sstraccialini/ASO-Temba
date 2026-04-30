import pickle
import os
import argparse

def convert_pkl_format(input_pkl, output_pkl):
    # Load original pickle file
    with open(input_pkl, 'rb') as f:
        original_data = pickle.load(f)
    
    # Convert format - extract only predictions
    converted_data = {}
    for video_id, data in original_data.items():
        if isinstance(data, dict):
            # Handle old format where data is a dictionary
            converted_data[video_id] = data['predictions'].squeeze(0)  # Remove batch dimension
        else:
            # Handle current format where data is already numpy array
            converted_data[video_id] = data.squeeze(0)  # Remove batch dimension
    
    # Save converted data
    with open(output_pkl, 'wb') as f:
        pickle.dump(converted_data, f)
    
    print(f"Converted file saved to: {output_pkl}")
    # Print shape of first item as example
    first_key = next(iter(converted_data))
    print(f"Example shape for video {first_key}: {converted_data[first_key].shape}")

def main():
    parser = argparse.ArgumentParser(description='Convert pickle file format to contain only predictions')
    parser.add_argument('--input_pkl', type=str, required=True, help='Path to input pickle file')
    parser.add_argument('--output_pkl', type=str, help='Path to output pickle file')
    
    args = parser.parse_args()
    
    # If output path not specified, create one based on input path
    if args.output_pkl is None:
        input_dir = os.path.dirname(args.input_pkl)
        input_filename = os.path.basename(args.input_pkl)
        output_filename = 'converted_' + input_filename
        args.output_pkl = os.path.join(input_dir, output_filename)
    
    convert_pkl_format(args.input_pkl, args.output_pkl)

if __name__ == '__main__':
    main()