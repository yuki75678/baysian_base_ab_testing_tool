from src.training.train_record import TrainRecord


def monte_carlo_simularer_ctr(
        test_model_record:TrainRecord,
        control_model_record:TrainRecord
        ):
    print(test_model_record.model_param)
    for i in test_model_record.model_param:
        print(i)
    
    win_rate_calclation_manager()

    difference_of_theta_manager()



def win_rate_calclation_manager():
    pass

def difference_of_theta_manager():
    pass

