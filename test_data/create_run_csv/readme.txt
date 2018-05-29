Benutzung:
    create_run_csv runsize label infile outfile
    
Beispiel:
    create_run_csv 200 1 out_0.txt runs_b.csv
    
    Zerschneidet out_0.txt in Blöcke von 200 Zeilen (die jeweils x- und y-Wert enthalten),
    verknüpft die x- und y-Werte mit ", " und hängt am ende das label 1 an. Speichert in runs_b.csv
    
    
    
Bauen:
    clang -o create_run_csv main.cpp