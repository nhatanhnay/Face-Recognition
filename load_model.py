def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0  # initialize counters for the matched and total weights
    curr_state_dict = model.state_dict()
    # get the current state dictionary of the model
    for key in curr_state_dict.keys():
        # iterate through all the keys in the model's state dictionary
        num_total += 1  # increment the total weight count
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            # check if the key exists in the given state dictionary and if the shapes of the weights match
            curr_state_dict[key] = state_dict[key]
            # assign the weight value from the given state dictionary to the current model's state dictionary
            num_matched += 1  # increment the matched weight count
    model.load_state_dict(curr_state_dict)
    # load the updated state dictionary to the model
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')
