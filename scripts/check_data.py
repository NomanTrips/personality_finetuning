import json

def check_bad_examples(file_path):
    bad_examples = []
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            data = json.loads(line)
            messages = data.get("messages", [])
            # Starting role should be 'user' based on provided examples and requirements
            expected_role = 'user'
            for message in messages:
                role = message.get("role")
                # Check if current message role is not as expected
                if role != expected_role:
                    bad_examples.append(idx)
                    break  # No need to check further messages in this example
                # Flip expected role for next iteration
                expected_role = 'assistant' if expected_role == 'user' else 'user'
    return bad_examples

# Replace 'your_file_path.jsonl' with the actual path to your JSONL file
file_path = './train.jsonl'
bad_examples = check_bad_examples(file_path)
if bad_examples:
    print(f"Found bad examples at lines: {bad_examples}")
else:
    print("No bad examples found.")
