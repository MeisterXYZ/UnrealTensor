runseq_size=200

training_runs_start=0
training_runs_end=4
training_csv_name="training.csv"

test_runs_start=5
test_runs_end=6
test_csv_name="test.csv"

names=("Raphael" "Dennis" "Carsten")
labels=(0 1 2)
number_of_persons=2

for k in {0..$number_of_persons}
do
    for r in {$training_runs_start..$training_runs_end}
    do
        infile_name = ${names[k]}"_out_"$r".txt"
        create_run_csv $runseq_size ${labels[k]} $infile_name $training_csv_name
    done
    
    for r in {$test_runs_start..$test_runs_end}
    do
        infile_name = ${names[k]}"_out_"$r".txt"
        create_run_csv $runseq_size ${labels[k]} $infile_name $test_csv_name
    done
done
