import hmm

states = ["-1", "-0.5", "0", "1", "2"] # States should be list of strings.
transition = {} # Transition is a dictionary of dictionaries.
for i in states:
    transition[i] = {}
    for j in states:
        if i == j:
            transition[i][j] = np.log(0.4)
        else:
            transition[i][j] = np.log(0.15)
emission = {} # Emission is dictionary of lists. each key is mean, variance.
for i in states:
    emission[i] = [float(i), 1] 
p_init_state = {i: np.log(1/len(states)) for i in states}


# Sequences passed in is a list of sequences. If just 1 sequence, just pass in a list with 1 sequence. 
seq = [merged_df['CPT000814'][merged_df['chromosome'] == '1'].to_list()] 

# This is how to use the code to get log likelihood.
state_path, transition, emission = hmm.ViterbiTraining(seq, states, transition, emission, p_init_state)
log_likelihood = hmm.LogLikelihood(seq, state_path, emission, transition, p_init_state)