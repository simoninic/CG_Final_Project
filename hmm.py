import os
import numpy as np
from scipy.special import logsumexp


# initialize emission probabilties matrix
def ViterbiTraining(
    sequences: list[str],
    emission_vals: list[str],
    state_vals: list[str],
    transition_mm: dict[str, dict[str, float]],
    emission_mm: dict[str, dict[str, float]],
    init_mm: dict[str, int],
) -> str:
    """
    Performs viterbi training determine state sequences.
    Input:
        seq: the input sequences.
        emission_vals: the list of possible emitted characters.
        state_vals: the list of possible state characters.
        transition_mm: A dictionary holding all the transition probabilities.
        emission_mm: A dictionary holding all the emission probabilities.
    Output:
        List of predicted state sequences for each input sequences.
        The transition probabilities.
        The emission probabilities.
    """
    current_path = ["" for _ in sequences]
    new_paths = [None for _ in sequences]
    while current_path != new_paths:
        current_path = new_paths
        for i, seq in enumerate(sequences):
            state_path = ViterbiDecoding(seq, emission_vals, state_vals, transition_mm, emission_mm, init_mm)
            new_paths[i] = state_path
        emission_mm = UpdateEmission(sequences, new_paths, state_vals, emission_vals)
        transition_mm = UpdateTransition(new_paths, state_vals)

    return current_path, transition_mm, emission_mm


def UpdateEmission(sequences: list[str], state_paths: list[str], state_vals: list[str]):
    """
    Creates an emission matrix based on a sequence of emitted characters and a sequence of states.
    Input:
        seq: the sequence of emitted characters
        state_path: the sequence of states
        state_vals: the possible states.
    Output:
        The updated emission probabiltiies.
    """


def UpdateEmission(sequences: list[str], state_paths: list[str], state_vals: list[str], emission_vals: list[str]):
    """
    Creates an emission matrix based on a sequence of emitted characters and a sequence of states.
    Input:
        seq: the sequence of emitted characters
        state_path: the sequence of states
        state_vals: the possible states.
    Output:
        The updated emission probabiltiies.
    """
    emission_count = {}
    for s in state_vals:
        emission_count[s] = {}
        for e in emission_vals:
            emission_count[s][e] = 1

    for seq, path in zip(sequences, state_paths):
        for char, state in zip(seq, path):
            emission_count[state][char] += 1

    for s in state_vals:
        total = 0
        for e in emission_vals:
            total += emission_count[s][e]
        for e in emission_vals:
            emission_count[s][e] /= total
            emission_count[s][e] = np.log(emission_count[s][e])
    return emission_count


def UpdateTransition(state_paths: list[str], state_vals: list[str]):
    """
    Creates an transition matrix based on a sequence of emitted characters and a sequence of states.
    Input:
        seq: the sequence of emitted characters
        state_path: the sequence of states
        state_vals: the possible states.
    Output:
        the updated transition probabilties.
    """
    transition_mm = {}
    for a in state_vals:
        transition_mm[a] = {}
        for b in state_vals:
            transition_mm[a][b] = 1

    for path in state_paths:
        for i in range(len(path) - 1):
            curr_state = path[i]
            next_state = path[i + 1]
            transition_mm[curr_state][next_state] += 1

    for a in state_vals:
        total = 0
        for b in state_vals:
            total += transition_mm[a][b]
        for b in state_vals:
            transition_mm[a][b] /= total
            transition_mm[a][b] = np.log(transition_mm[a][b])
    return transition_mm


def ViterbiDecoding(
    seq: str,
    emission_vals: list[str],
    state_vals: list[str],
    transition_mm: dict[str, dict[str, float]],
    emission_mm: dict[str, dict[str, float]],
    init_prob: dict[int, int],
) -> str:
    """
    Uses the Viterbi algorithm to determine the sequence of states for an input sequence.
    Input:
        seq: The input sequence.
        state_0: The initial probabilities of each state.
        transitions: transition probabilities
        emissions: emission probabilities
    Output:
        Sequence of states.
    """

    score_matrix = np.zeros([len(state_vals), len(seq)])
    path_matrix = np.zeros(score_matrix.shape) - 1  # Initialize a matrix of -1. 0 = vampire, 1 = werewolf
    for row, state in enumerate(state_vals):
        score_matrix[row, 0] = init_prob[state] + emission_mm[state][seq[0]]

    for col in range(1, score_matrix.shape[1]):
        for row in range(score_matrix.shape[0]):
            new_state = state_vals[row]
            best_score = -np.inf
            best_prev_state = -1
            for i, old_state in enumerate(state_vals):  # Number of states
                score = (
                    score_matrix[i, col - 1] + transition_mm[old_state][new_state] + emission_mm[new_state][seq[col]]
                )
                if score > best_score:
                    best_score = score
                    best_prev_state = i

            score_matrix[row, col] = best_score
            path_matrix[row, col] = best_prev_state

    start_idx = int(score_matrix[:, -1].argmax())
    path = Backtrack(path_matrix, state_vals, start_idx)
    return path


def Backtrack(path_matrix: np.ndarray, state_vals: list[str], start_idx: int) -> str:
    """
    Generates the path of states given the path matrix.
    Input:
        path_matrix: matrix containing path data.
        start_idx: the starting row in the last column of path_matrix.
    Output:
        the path as a string.
    """
    current_idx = start_idx
    path = ""
    for i in range(path_matrix.shape[1] - 1, -1, -1):
        path = "".join([state_vals[current_idx], path])
        current_idx = int(path_matrix[current_idx, i])
    return path


def CountLetterDifferences(s1: str, s2: str) -> int:
    """
    Count the number of different letters between two strings element wise.
    Input:
        s1, s2: the input strings
    Output:
        The number of differing characters.
    """
    if len(s1) != len(s2):
        raise Exception("Length of sequences are not the same.")

    count = 0
    for v, e in zip(s1, s2):
        if v != e:
            count += 1
    return count


def LogLikelihood(sequences, state_paths, emissions, transitions, init_probs):
    log_likelihood = 0
    for sequence, state_path in zip(sequences, state_paths):
        prev_state = None
        for idx, (char, state) in enumerate(zip(sequence, state_path)):
            if idx == 0:
                log_likelihood += emissions[state][char] + init_probs[state]
            else:
                log_likelihood += emissions[state][char] + transitions[prev_state][state]
            prev_state = state
    return log_likelihood


def GetAllDistributions(seqs):
    all_distributions = []
    for seq in seqs:
        all_distributions.append(DetermineDistributionOfSeq(seq))
    return all_distributions


def DetermineDistributionOfSeq(seq):
    counter = {}
    for val in seq:
        if val in counter:
            counter[val] += 1
        else:
            counter[val] = 1
    for val in counter:
        counter[val] = counter[val] / len(seq)
    return counter


def ConvertToAmpLoss(data):
    new_data = np.zeros(len(data))
    for i, val in enumerate(data):
        if val < -0.2:
            new_data[i] = -1
        if val > 0.2:
            new_data[i] = 1
        else:
            new_data[i] = 0
    return new_data


####

def BaumWelch(seq, transition, emission, weights):
    states = sorted(list(transition.keys()))
    loglikelihood = -np.inf
    num_iter = 0
    delta = np.inf
    while np.abs(delta) > 1 and num_iter < 50:
        num_iter += 1
        old_loglikelihood = loglikelihood
        alpha, beta, gamma, epsilon = EStep(seq, transition, emission, weights)
        transition, emission, weights = MStep(seq, gamma, epsilon, states)
        loglikelihood = logsumexp(alpha[:, -1])
        delta = loglikelihood - old_loglikelihood
        print(delta)

    return loglikelihood

def LogNormal(x: float, mu: float, sigma: float):
    return -np.log(sigma * np.sqrt(2 * np.pi)) - 0.5 * (np.power((x - mu) / sigma, 2))

def EStep(
    seq: list[float],
    transition_matrix: dict[str, dict[str, int]],
    emission_matrix: dict[str, list[int]],
    p_state: dict[str, int],
):
    """
    E-Step of Baum-Welch algorithm.
    """
    alpha_matrix = Forward(seq, transition_matrix, emission_matrix, p_state)
    beta_matrix = Backward(seq, transition_matrix, emission_matrix)
    gamma = CalculateProbabiiltyMatrix(alpha_matrix, beta_matrix)
    epsilon = CalculateJointProbability(seq, alpha_matrix, beta_matrix, transition_matrix, emission_matrix)

    # Normalize. This is probably wrong but i'm doing it anyways...
    gamma = gamma - logsumexp(gamma, axis = 0)
    epsilon = epsilon - logsumexp(epsilon, axis = 1) + gamma[:, :-1]
    return alpha_matrix, beta_matrix, gamma, epsilon

def MStep(seq: list[float], gamma: np.ndarray, epsilon: np.ndarray, states: list[str]) -> tuple[dict[str, dict[str, int]], dict[str, list[int]]]:
    """
    M-Step of Baum-Welch algorithm.
    """
    transition = UpdateTransition(gamma, epsilon, states)
    emission = UpdateEmission(seq, gamma, states)
    weights = UpdateWeights(gamma)
    return transition, emission, weights

def UpdateWeights(gamma):
    weights = logsumexp(gamma, axis = 1)
    weights -= logsumexp(weights)
    return weights

def UpdateTransition(gamma: np.ndarray, epsilon: np.ndarray, states: list[str]):
    """
    Updates the transition probability matrix.
    Input:
        gamma: Log(S(i)). A n x len(sequence) matrix. Where n is the number of states
        epsilon: log(S(i, j)). A n x n x len(sequence) matrix. Where n is the number of states.
        states: a list of possible states. Should be sorted.
    """
    transition = {}
    for i, s1 in enumerate(states):
        transition[s1] = {}
        for j, s2 in enumerate(states):
            transition[s1][s2] = logsumexp(epsilon[i, j, :])
            transition[s1][s2] -= logsumexp(gamma[i])
    return transition

def UpdateEmission(seq: list[float], gamma: np.ndarray, states: list[str]):
    """
    Updates the emission probability matrix.
    seq: Sequence of observations.
    gamma: (S(i)). A n x len(sequence) matrix. Where n is the number of states.
    states: a list of possible states. Should be sorted.
    """
    gamma = np.exp(gamma)
    emission = {}
    for i, s1 in enumerate(states):
        new_mean = np.sum(seq * gamma[i, :])
        new_mean /= np.sum(gamma[i, :])    

        mean_center = seq-new_mean
        new_cov = np.sum(mean_center * mean_center * gamma[i, :])
        new_cov /= np.sum(gamma[i,:])

        emission[s1] = [new_mean, new_cov]
    return emission

def CalculateJointProbability(
    seq: list[float],
    alpha_matrix: np.ndarray[float],
    beta_matrix: np.ndarray[float],
    transition_matrix: dict[str, dict[str, float]],
    emission_matrix: dict[str, list[float]],
    # t: int,
    # state_t: int,
    # state_t1: int,
):
    states = sorted(list(transition_matrix.keys()))
    state_to_idx = {}
    for i, key in enumerate(states):
        state_to_idx[key] = i
    seq_length = alpha_matrix.shape[1]

    # Set up probability matrix
    prob_matrix = np.zeros((len(states), len(states), seq_length-1))
    for i, s in enumerate(states):
        prob_matrix[i, :, :] = alpha_matrix[:, :-1] + beta_matrix[state_to_idx[s], 1:]
    
    # Add log emission prob
    for i, s in enumerate(states):
        mu, sigma = emission_matrix[s]
        for j, obs in enumerate(seq[1:]):
            emit = LogNormal(obs, mu, sigma)
            prob_matrix[:, i, j] += emit
    
    # Add log transition prob
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            prob_matrix[i, j, :] += transition_matrix[s1][s2]

    prob_matrix -= logsumexp(prob_matrix)
    return prob_matrix


def Forward(
    seq: list[int],
    transition_matrix: dict[str, dict[str, int]],
    emission_matrix: dict[str, list[int]],
    p_state: dict[str, int],
):
    """
    Calculates the forwards probability in Baum-Welch algorithm.
    Input:
        seq: The sequence of observations.
        transition_matrix: The current transition probabilities.
        emissione_matrix: The current emission probabilities.
        p_state: The probability of a state.
    Output:
        The log forward matrix in Baum-Welch algorithm.
    """
    states = sorted(list(transition_matrix.keys()))
    state_to_idx = {}
    for i, key in enumerate(states):
        state_to_idx[key] = i

    alpha_matrix = np.zeros((len(states), len(seq)))

    for i, state in enumerate(states):
        mu, sigma = emission_matrix[state]
        emit = LogNormal(seq[0], mu, sigma)
        alpha_matrix[i, 0] = emit + p_state[state_to_idx[state]]

    for t in range(1, len(seq)):
        obs = seq[t]
        for i, state in enumerate(states):
            alpha_matrix_cell = [
                0 for _ in range(len(states))
            ]  # List to hold all the log probabilities so we can do logsumexp
            for j, prev_state in enumerate(states):
                alpha_matrix_cell[j] = transition_matrix[prev_state][state] + alpha_matrix[j, t - 1]

            mu, sigma = emission_matrix[state]
            emit = LogNormal(obs, mu, sigma)
            alpha_matrix[i, t] = logsumexp(alpha_matrix_cell) + emit

    return alpha_matrix


def Backward(seq: list[int], transition_matrix: dict[str, dict[str, int]], emission_matrix: dict[str, list[int]]):
    """
    Calculates the backwards probability in Baum-Welch algorithm.
    Input:
        seq: The sequence of observations.
        transition_matrix: The current transition probabilities.
        emissione_matrix: The current emission probabilities.
    Output:
        The log backward matrix in Baum-Welch algorithm.
    """
    states = sorted(list(transition_matrix.keys()))
    state_to_idx = {}
    for i, key in enumerate(states):
        state_to_idx[key] = i

    beta_matrix = np.zeros((len(states), len(seq)))

    for i, state in enumerate(states):
        beta_matrix[i, len(seq) - 1] = 0

    for t in range(len(seq) - 2, -1, -1):
        obs = seq[t + 1]
        for i, state in enumerate(states):
            beta_matrix_cell = [0 for _ in range(len(states))]
            for j, next_state in enumerate(states):
                mu = emission_matrix[next_state][0]
                sigma = emission_matrix[next_state][1]
                emit = LogNormal(obs, mu, sigma)
                beta_matrix_cell[j] = transition_matrix[state][next_state] + emit + beta_matrix[j, t + 1]
                # beta_matrix[i, t] += (
                #     transition_matrix[state][next_state] * emission_matrix[next_state][obs] * beta_matrix[j, t + 1]
                # )
            beta_matrix[i, t] = logsumexp(beta_matrix_cell)

    return beta_matrix


def CalculateProbabiiltyMatrix(alpha_matrix, beta_matrix):
    """
    Calculates gamma_i(t) which is the probability of being in state t at time i given the parameters theta and the
    observed sequence.
    """
    prob_matrix = alpha_matrix + beta_matrix
    prob_matrix -= np.apply_along_axis(logsumexp, axis = 0, arr = prob_matrix)[np.newaxis, :]
    # prob_matrix = prob_matrix / np.sum(prob_matrix, axis=0)
    return prob_matrix
