#!/bin/bash

folder_path=$1

min_fid=999999.0
min_fid_file=""

# Use process substitution instead of pipe to avoid subshell
while read -r file; do
    if [ -r "$file" ]; then
        fid=$(grep -A1 "FID:" "$file" | tail -2 | awk '{if($1=="FID:") print $2}' | tr -d '[:space:]')
        echo "FID value for $file: $fid"

        comparison=$(echo "$fid < $min_fid" | bc -l)
        if [ "$comparison" -eq 1 ]; then
            min_fid=$fid
            min_fid_file=$file
        fi

        echo "Current minimum: $min_fid"
    else
        echo "Warning: Cannot read file: $file (Permission denied)"
    fi
done < <(find "$folder_path" -type f -name "*.txt")

echo "Final Minimum FID value: $min_fid"
echo "Filepath of the file with minimum FID value: $min_fid_file"