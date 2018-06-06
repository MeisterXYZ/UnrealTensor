for run in {1..8}
	do
    let perceptNos=$run*100
	python train_and_test.py runs2.csv validRuns2.csv $perceptNos $perceptNos $perceptNos
	done