runseq_size=200

runs_start=0
runs_end=6
csv_name="data__"$runseq_size".csv"

names=("Raphael" "Dennis" "Carsten")
labels=(0 1 2)
number_of_persons=3

for ((k=0;k<number_of_persons;k++))
do
    for ((r=runs_start;r<=runs_end;r++))
    do
        infile_name=${names[k]}"_out_"$r".txt"
        create_run_csv $runseq_size ${labels[k]} $infile_name $csv_name
    done
done
