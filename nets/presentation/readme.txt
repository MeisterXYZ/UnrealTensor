To run script on commandline:
    python neural_net.py runs_1_to_6_slices_200.csv predict_dennis_run_7.csv 800 4 200 100
    
where
    neural_net.py                   name of the script
    
    runs_1_to_6_slices_200.csv      name of dataset containing training and test data
                                    80% of the data gets chosen randomly to be assigned
                                    to the training set, the rest gets assigned to test set
    
    predict_dennis_run_7.csv        data to be predicted by the fully trained net 
                                    (the net does not see the original labels)
                                    
    800                             number of neurons per hidden layers
    
    4                               number of hidden layers
    
    200                             number of (x, y)-mousedelta tuples per input tensor
    
    100                             number of epochs (used for training)