Benutzung:
    create_run_csv runsize label infile outfile
    
Beispiel:
    create_run_csv 200 1 out_0.txt runs_b.csv
    
    Zerschneidet out_0.txt in Blöcke von 200 Zeilen (die jeweils x- und y-Wert enthalten),
    verknüpft die x- und y-Werte mit ", " und hängt am ende das label 1 an. Speichert in runs_b.csv
    
    
    
Bauen:
    clang -o create_run_csv main.cpp
Mac:
    clang++ -o create_run_csv create_run_csv.cpp


sh-command:
(./create_run_csv 100 0 Raphael_out_0.txt runs1.csv ; ./create_run_csv 100 0 Raphael_out_1.txt runs1.csv ; ./create_run_csv 100 0 Raphael_out_2.txt runs1.csv ; ./create_run_csv 100 0 Raphael_out_3.txt runs1.csv ; ./create_run_csv 100 0 Raphael_out_4.txt runs1.csv; ./create_run_csv 100 1 Dennis_out_0.txt runs1.csv ; ./create_run_csv 100 1 Dennis_out_1.txt runs1.csv ; ./create_run_csv 100 1 Dennis_out_2.txt runs1.csv ; ./create_run_csv 100 1 Dennis_out_3.txt runs1.csv ; ./create_run_csv 100 1 Dennis_out_4.txt runs1.csv)

(./create_run_csv 100 0 Raphael_out_5.txt validRuns1.csv ;./create_run_csv 100 0 Raphael_out_6.txt validRuns1.csv ;./create_run_csv 100 1 Dennis_out_5.txt validRuns1.csv ;./create_run_csv 100 1 Dennis_out_6.txt validRuns1.csv ;)
    