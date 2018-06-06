runseq_size=500

training_runs_start=0
training_runs_end=4
training_csv_name="training_"$runseq_size".csv"

test_runs_start=5
test_runs_end=6
test_csv_name="test_"$runseq_size".csv"

names=("Raphael" "Dennis" "Carsten")
labels=(0 1 2)
number_of_persons=3

for ((k=0;k<number_of_persons;k++))
do
    for ((r=training_runs_start;r<=training_runs_end;r++))
    do
        infile_name=${names[k]}"_out_"$r".txt"
        create_run_csv $runseq_size ${labels[k]} $infile_name $training_csv_name
    done
    
    for ((r=test_runs_start;r<=test_runs_end;r++))
    do
        infile_name=${names[k]}"_out_"$r".txt"
        create_run_csv $runseq_size ${labels[k]} $infile_name $test_csv_name
    done
done
