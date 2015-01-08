# python functions related to file loss probability calculation.

import utilities

from itertools import (cycle, islice, combinations, chain)
import numpy as np
from math import (factorial)

def window(seq, n):
    """
      Implement a sliding window of size n for a sequence seq.

    Args: 
        seq (list): The sequence we want to apply the sliding window.
        n   (int): The size of the sliding window.

    Returns:
           Returns a sliding window as a tuple.
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def feasible_configurations(disk_count, chunk_count, spread_factor):
    """
      Compute feasible configurations.
    
    Args:
        disk_count (int): number of disks in the system.
        chunk_count (int): number of chunks the file is spread into.
        spread_factor (int): number of disks to be used after selected disk.
        
    Returns:
        Set of feasible configurations represented as a tuple of integers. 
    """
    remaining_chunks = chunk_count - 1
    selected_disks = spread_factor + 1
    
    disk_ring = cycle(range(disk_count))
    
    for i, w in enumerate(window(disk_ring, selected_disks)):
        main_disk, consecutive_disks = w[0], islice(w, 1, None)
        for c in combinations(consecutive_disks, remaining_chunks):
            yield (main_disk,) + c
        if i >= disk_count - 1:
            break

def disk_loss_probability(disks, reliability, k):
    return sum([product([1 - reliability[disk] for disk in failing_disks]) *
                product([reliability[disk] for disk in disks if disk not in failing_disks])
                for failing_disks in combinations(disks, k)])

def probability_at_least_d_fail(disks, d, chunk_count, reliability):
    """Compute the probability of observing at least d disks failure for a special configuration.
    
    Args:
        disks (int: list) : disk indexes in {1,...,m} where chunks are stored.
        d (int) : number of loss from which a file is lost.
        chunk_count (int): number of chunks the file is spread into.
        reliability (int: double): list of chunk_count real in ]0;1[.
        
    Returns:
        The probability of observing at least d disks failure for a special configuration as a real. 
    """
    #print(disks, d, chunk_count, reliability)
    
    return sum([disk_loss_probability(disks, reliability, k) for k in range(d, chunk_count + 1)])
    # return sum([disk_loss_probability(k) for k in range(recovery_threshold, chunk_count + 1)])

def transfer_matrix(reliability, size):
    """
    Construct square transfer matrix used.
    
    Args:
        reliability (double) : a probability in ]0,1[.
        size (int) : The size of the transfer matrix.
        
    Returns:
        A (d+1)x(d+1) matrix with non-zero elements only on the diagonal and on the upper diagonal, as a sparse array.
    """

    return ((1.0 - reliability) * np.eye(size, k=0, dtype=float) +
            reliability * np.eye(size, k=1, dtype=float))

def probability_at_least_d_fail_matrix(disks, d, chunk_count, reliability):
    """Compute the probability of observing at least d disks failure for a special configuration using matrix formulation.
    
    Args:
        configuration (int) : disk indexes in {1,...,m} where chunks are stored.
        d (int) : number of loss from which a file is lost.
        chunk_count (int): number of chunks the file is spread into.
        reliability (double): list of chunk_count real in ]0;1[.
        
    Returns:
        The probability of observing at least d disks failure for a special configuration as a real. 
    """
    failure_threshold = chunk_count - d + 1
    
    transfer_matrices = [transfer_matrix(reliability[disk], failure_threshold)
                         for disk in reversed(disks)]
    
    return np.sum(reduce(np.dot, transfer_matrices), axis=1)[0]

def loss_probability(reliability, k, chunk_count):
    """Compute the probability of exactly k disks faile
    
    Args:
        reliability (double): The reliability of disks.
        k (int): The number of failing disks.
        chunk_count (int): The number of chunks the file is psread into.
        
    Returns:
        The probability of losing exactly k disks as a double.
    """
    return (factorial(chunk_count)/(factorial(chunk_count-k)*factorial(k))
            * pow(1 - reliability,k)*pow(reliability,chunk_count-k))
    
def probability_at_least_d_fail_equal_reliability(d, chunk_count, reliability):
    """Compute the probability of observing at least d disks failure for a special configuration if all reliabilities are equal.
    
    Args:
        d (int) : number of loss from which a file is lost.
        chunk_count (int): number of chunks the file is spread into.
        reliability (double): disks reliability.
        
    Returns:
        The probability of observing at least d disks failure for a special configuration as a real. 
    """
    
    return sum([loss_probability(reliability, k, chunk_count) for k in range(d, chunk_count + 1)])

def distance(j, k, disk_count):
    """Compute the distance between i and j as explain above.
    
    args:
        j (int): the index of the first disk selected to sotre the first chunk.
        k (int): the index of an other disk that the disk j in the configuration.
        disk_count (in): the number of disk in the server.
        
    Returns:
        The distance between j and k as an integer.
    """
    
    if j <= k:
        return k - j
    else:
        return distance(j, disk_count, disk_count) + k
    
def list_distances(configuration, disk_count, j_index ):
    """Compute distances between j and the other elements in the configuration.
    
    args:
        configuration (list): list of index between {1, ... , chunk_count}.
        disk_count (int): number of disk in the server.
        j_index  (int): the index of the first disk selected.
        
    Returns:
        List of distances as a list of integers.
    """
    
    for other_index in list(set(configuration)-set([j_index])):
        yield distance(j_index, other_index, disk_count)
            
def frequency(configuration, disk_count, spread_factor):
    """Compute the frequency a configuration can be selected.
    
    args:
        configuration (list): list of index between {1, ... , chunk_count}.
        disk_count (int): number of disk in the server.
        spread_factor (int): the spread_factor.
        
    Returns:
        Number of way we have to create the configuration as an integer.
    """
    
    way_count = 0
    
    for index in configuration:
        minS_j = max([element for element in list_distances(configuration, disk_count, index )])
        if spread_factor >= minS_j:
            way_count += 1
            
    return way_count
  
def probability_file_loss(disk_count, chunk_count, spread_factor, d, disk_reliability):
    """Compute the probability of file loss.
    
    Args:
        disk_count (int): number of disks in the system.
        chunk_count (int): number of chunks the file is spread into.
        spread_factor (int): number of disks to be used after selected disk.
        disk_reliability (double): list of chunk_count real in ]0;1[.
        
    Returns:
        The probability that a file is lost as a real. 
    """
    # First we calculate the probability that a configuration is selected.
    probability_configuration = (1.0*(factorial(spread_factor - chunk_count + 1)*factorial(chunk_count - 1))
                              / (disk_count*factorial(spread_factor)))
    
    # Then, we generate the list of all feasible configurations as a function of 
    # the number of disks, disk_count, the number of chunks, chunk_count and the spread factor.
    #configurations = list(map(lambda t: map(lambda e: e + 1, t),
    #    feasible_configurations(disk_count, chunk_count, spread_factor)))
  
    configurations = feasible_configurations(disk_count, chunk_count, spread_factor)

    # Finally, for each feasible configuration where chunks are stored, we calculate the probabi-
    # lity the file is lost.
    
    # first we consider there is no redondancy
    if disk_count >= 2*(spread_factor + 1) - chunk_count:
        return probability_configuration * sum([probability_at_least_d_fail(config, d, chunk_count, disk_reliability)
                                                for config in configurations])
    # This condition cover the case with redondancies
    else:
        return probability_configuration * sum([frequency(config, disk_count, spread_factor)*probability_at_least_d_fail(config, d, chunk_count, disk_reliability)
                                               for config in configurations])

def probability_file_loss_matrix(disk_count, chunk_count, spread_factor, d, disk_reliability):
    """Compute the probability of file loss using a matriw formulation.
    
    Args:
        disk_count (int): number of disks in the system.
        chunk_count (int): number of chunks the file is spread into.
        spread_factor (int): number of disks to be used after selected disk.
        disk_reliability (double): list of chunk_count real in ]0;1[.
        
    Returns:
        The probability that a file is lost as a real. 
    """
    
    # First we calculate the probability that a configuration is selected.
    probability_configuration = (factorial(spread_factor - chunk_count + 1) * factorial(chunk_count - 1) /
                                 (disk_count * factorial(spread_factor)))
    
    # Then, we generate the list of all feasible configurations as a function of 
    # the number of disks, disk_count, the number of chunks, chunk_count and the spread factor.
    configurations = feasible_configurations(disk_count, chunk_count, spread_factor)
  
    # Finally, for each feasible configuration where chunks are stored, we calculate the probabi-
    # lity the file is lost.
    
    # first we consider there is no redundancy
    if disk_count >= 2*(spread_factor + 1) - chunk_count:

        return sum([probability_configuration * 
                    probability_at_least_d_fail_matrix(config, d, chunk_count, disk_reliability)
                    for config in configurations])
    # This condition cover the case with redundancies
    else:
        return sum([frequency(config, disk_count, spread_factor) *
                    probability_configuration *
                    probability_at_least_d_fail_matrix(config, d, chunk_count, disk_reliability)
                    for config in configurations])

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
        
        reliability = dict(zip(disks, uniform(mean_reliability-0.5*delta,mean_reliability+0.5*delta, size=len(disks))))
        reliability[disks[-1]] = len(disks) * mean_reliability - sum([reliability[disk] for disk in disks[:-1]])
        
        # Then we calculate the probability of file loss for the reliability list 
        yield probability_file_loss(len(disks), chunk_count, spread_factor, d, reliability)
