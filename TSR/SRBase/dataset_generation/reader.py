import os
import json

# Default round value for probability approximation
ROUND_VALUES = 6

def prob_string_to_dict(str):
    # Get probability dict information from string in dataset_configuration.json
    pairs = str.split(",")
    result_dict = {}
    for pair in pairs:
        key, value = pair.split(":")
        result_dict[key] = float(value) / 100
    return result_dict

def compute_each_ops_prob(prob_dict):
    # Compute probability of each operation
    operation_prob = {}
    for key, inner_dict in ((k, v) for k, v in prob_dict.items() if not k.startswith("func_type")):
        index = key.find("_prob")
        func_type_key = key[:index]
        for inner_key, inner_value in inner_dict.items():
            operation_prob[inner_key] = inner_value * prob_dict["func_type_prob"][func_type_key]
    operation_prob = {key: round(value, ROUND_VALUES) for key, value in operation_prob.items()}
    return operation_prob
    
def get_dataset_configuration():
    # Get structurized dataset configuration for dataset generation
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, 'dataset_configuration.json')
    with open(file_path, 'r') as file:
        data_dict = json.load(file)

    prob_dict = {key: prob_string_to_dict(value) for key, value in data_dict.items() if key.endswith("_prob")}
    for key, inner_dict in prob_dict.items():
        assert round(sum(inner_dict.values()),ROUND_VALUES) == 1, "Sum of probability of each type of operations should be 1.000, but got %.3f in %s" % (sum(inner_dict.values()), str(key))

    operation_prob = compute_each_ops_prob(prob_dict)
    assert round(sum(operation_prob.values()),ROUND_VALUES) == 1, "Sum of probability of operations should be 1.000, but got %.3f in %s" % (sum(operation_prob.values()), str(key))

    args_generate_datasets = {key: value for key, value in data_dict.items() if not key.endswith("_prob")}
    args_generate_datasets['operation_prob'] = operation_prob

    return args_generate_datasets

if __name__ == '__main__':
    cfg = get_dataset_configuration()
    print(cfg)