#!/usr/bin/env python3
"""
main.py
Main entry point for the video processing and security detection system.
This script initializes and runs the processing pipeline with the specified
video source and output options.
Author: fw7th
Date: 2025-04-26
"""

import argparse
from data import Compile

def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value  # return as string if not int

def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(description='Video processing and security detection system')

    parser.add_argument('--source', '-s', type=int_or_str, default=0,
                        help='Video source (file path, RTSP URL, or camera index)')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path for processed video')
    
    parser.add_argument('--enable-saving', action='store_true',
                        help='Enable saving of processed video')
    
    parser.add_argument('--phone', type=str, default=None,
                        help='Phone number for SMS alerts')
    
    parser.add_argument('--email', type=str, default=None,
                        help='Email address for email alerts')

    parser.add_argument(
        '--accuracy', type=int_or_str, 
        choices=["low", "high", "mid", 1, 2, 3], default=None,
        help='Accuracy-Speed choice for inference')
    
    return parser.parse_args()

def main():
    """Main function to run the video processing pipeline."""
    args = parse_arguments()

    # Create and run the pipeline
    pipe = Compile(
        source=args.source,
        enable_saving=args.enable_saving,
        save_dir=args.output,
        your_num=args.phone,
        your_mail=args.email,
        accuracy=args.accuracy
    )
    pipe.run()

if __name__ == "__main__":
    main()
