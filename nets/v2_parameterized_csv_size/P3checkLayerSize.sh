for i in {3..3}
	do
    let seqSiz=$i*100
	for ((k=1500;k<=5000;k=k+500))
		do
		let perceptNos=$k
		echo "python3 train_and_test.py training_$seqSiz.csv test_$seqSiz.csv $perceptNos $perceptNos $seqSiz"
		python3 train_and_test.py training_$seqSiz.csv test_$seqSiz.csv $perceptNos $perceptNos $seqSiz
		done
	done