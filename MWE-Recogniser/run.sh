#!/bin/bash

echo "=== MWE Recognition Tools ==="
echo "Please select which method to run:"
echo "1. PMI and Chi Square Analysis"
echo "2. LVC, MVC Tagging"
read -r -p "Enter your choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    # PMI and Chi Square Analysis
    read -r -p "Enter the input file path: " input_file
    
    if [ ! -f "$input_file" ]; then
        echo "Error: Input file does not exist."
        exit 1
    fi
    
    read -r -p "Enter the number of top bigrams to display [default: 30]: " top_k
    # Set default value if empty
    top_k=${top_k:-30}
    
    echo "Running PMI and Chi Square analysis on $input_file (showing top $top_k bigrams)..."
    python scripts/pmi.py "$input_file" "$top_k"

elif [ "$choice" == "2" ]; then
    # LVC, MVC Tagging
    read -r -p "Enter the input file path: " input_file
    read -r -p "Enter the output file path: " output_file
    
    if [ ! -f "$input_file" ]; then
        echo "Error: Input file does not exist."
        exit 1
    fi
    
    # Create intermediate filename based on input
    input_base=$(basename "$input_file")
    input_dir=$(dirname "$input_file")
    intermediate_file="${input_dir}/${input_base%.*}-cause.conllu"
    
    echo "Step 1: Adding causative features to $input_file..."
    python scripts/feature_cause.py "$input_file" "$intermediate_file"
    
    if [ $? -ne 0 ]; then
        echo "Error: Feature extraction failed."
        exit 1
    fi
    
    echo "Step 2: Identifying multi-word expressions..."
    python scripts/vmwe.py "$intermediate_file" "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "Process completed successfully!"
        echo "Output file saved to: $output_file"
    else
        echo "Error: MWE tagging failed."
        exit 1
    fi

else
    echo "Invalid choice. Please run the script again and select 1 or 2."
    exit 1
fi
