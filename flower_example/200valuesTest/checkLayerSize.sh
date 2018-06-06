for run in {1..8}
	do
    let perceptNos=$run*100
	python3 tensor200Test.py runs1_b.csv validRuns1_b.csv $perceptNos $perceptNos
	done