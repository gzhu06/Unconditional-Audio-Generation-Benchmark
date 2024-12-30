
batch_folder=$1
folder_name=$(basename $batch_folder)

# output to a text file naming after batch folder
for folder in $batch_folder/*; do
    if [ -d "$folder" ]; then
        echo "Processing $folder ----------------------------------------------------------"
        
        # get the folder name
        exp_name=$(basename $folder)
        mkdir -p ./$folder_name/$exp_name

        # get the output file name
        output_file=./$folder_name/$exp_name/result.txt

        python test_speech_commands.py --sample-dir $folder/tensorboard/test_samples resnext.pth > $output_file

    fi
done

