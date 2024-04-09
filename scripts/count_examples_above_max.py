import json

def count_examples_over_max_length(file_path, max_length):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            example = json.loads(line)
            text_length = len(example['text'])
            
            if text_length > max_length:
                count += 1
    
    return count

# Specify the path to your JSONL file
file_path = './train.jsonl'

# Set the maximum length threshold
max_length = 2225

# Count the number of examples exceeding the max length
num_examples_over_max_length = count_examples_over_max_length(file_path, max_length)

print(f"Number of examples exceeding the max length of {max_length}: {num_examples_over_max_length}")