# This file contains function related to the analysis of the approximation of the file loss probability


#############################################################################################################################
#                                                       IMPORTS                                                             #
#############################################################################################################################
from file_loss import (probability_file_loss, probability_file_loss_matrix, 
                       probability_at_least_d_fail_equal_reliability)

from pandas import DataFrame
from numpy.random import (uniform)
#############################################################################################################################


#############################################################################################################################
#                                           FILE LOSS PROBABILITY APPROXIMATION                                             #
#############################################################################################################################
def file_loss_delta(delta, mean_reliability, experiments, disks, chunk_count, spread_factor, d):
    """To do.
    
    Args:
        delta (double): 
        mean_reliability:
        experience_count (int):
        disk_count (int): number of disks in the system.
        chunk_count (int): number of chunks the file is spread into.
        spread_factor (int): number of disks to be used after selected disk.
        d (int): 
        
    Returns:
        To do. 
    """
    
    for e in experiments:
        # First we generate the list of disk reliability between mean_reliability-0.5*delta and mean_reliability-0.5*delta.
        
        lower_bound = mean_reliability - 0.5 * delta
        upper_bound = mean_reliability + 0.5 * delta

        reliability = dict(zip(disks, uniform(low=lower_bound, high=upper_bound, size=len(disks))))
        reliability[disks[-1]] = len(disks) * mean_reliability - sum([reliability[disk] for disk in disks[:-1]])
        
        # Then we calculate the probability of file loss for the reliability list 
        yield probability_file_loss(disks, chunk_count, spread_factor, d, reliability)

def file_loss_delta_matrix(delta, mean_reliability, experiments, disks, chunk_count, spread_factor, d):
    """To do.
    
    Args:
        delta (double): 
        mean_reliability:
        experience_count (int):
        disk_count (int): number of disks in the system.
        chunk_count (int): number of chunks the file is spread into.
        spread_factor (int): number of disks to be used after selected disk.
        d (int): 
        
    Returns:
        To do. 
    """
    
    for e in experiments:
        # First we generate the list of disk reliability between mean_reliability-0.5*delta and mean_reliability-0.5*delta.

        lower_bound = mean_reliability - 0.5 * delta
        upper_bound = mean_reliability + 0.5 * delta

        reliability = dict(zip(disks, uniform(low=lower_bound, high=upper_bound , size=len(disks))))
        reliability[disks[-1]] = len(disks) * mean_reliability - sum([reliability[disk] for disk in disks[:-1]])
        
        # Then we calculate the probability of file loss for the reliability list 
        yield probability_file_loss_matrix(disks, chunk_count, spread_factor, d, reliability)

def launch_experiment(disks, chunk_count, threshold_recovery, spread_factor, experiment, delta, mean_reliability):
    """
    Args:
        disks (list):
        chunk_count (int):
        threshold_recovery (int):
        spread_factor (int):
        experiences (list):
        delta (double):
        mean_reliability (double):
        
    Returns:
    """
    
    # Approximated file loss probability
    reliability_mean_value = probability_at_least_d_fail_equal_reliability(threshold_recovery, chunk_count, mean_reliability)
    print("----------------------------------------------------------------------")
    print("                  Approximated file loss probability                  ")
    print(reliability_mean_value)
    print("----------------------------------------------------------------------")
    
    # Exact reliability dataframe
    print("----------------------------------------------------------------------")
    print("                     Exact file loss probability                     ")
    reliabilities = DataFrame(file_loss_delta_matrix(delta, mean_reliability, experiment, disks, chunk_count, spread_factor, 
                                                        threshold_recovery)
                              , columns=['Reliability'])
    
    # Histogram of reliabilities
    reliabilities.hist(bins=50)
    
    # Mean value and standard deviation of the dataframe reliabilities
    print "Mean value of reliabilities : ",reliabilities.mean()
    print("")
    print "Standard deviation of reliabilities : ",reliabilities.std()

    # relative error
    relative_error_list = abs(reliabilities - reliability_mean_value)/abs(reliabilities)
    
    # absolute error
    absolute_error_list = abs(reliabilities - reliability_mean_value)

    # relative error plot
    relative_error_list.plot(title='Relative erros plot')
    
    # relative error histogram
    relative_error_list.hist(bins=50)

    # absolute error plot
    absolute_error_list.plot(title='Absolute errors plot')
    
    # absolute error histogram
    absolute_error_list.hist(bins=50)
    
    print("----------------------------------------------------------------------")
#############################################################################################################################
