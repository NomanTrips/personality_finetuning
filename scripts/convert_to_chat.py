import json

def parse_conversation(text):
    """
    Parse the conversation, ensuring each message, whether from the user or the assistant,
    is correctly separated and tagged, including handling multiple turns within the conversation.
    """
    segments = text.split("<s>")
    messages = []

    for segment in segments:
        parts = segment.split("[INST]")
        for i, part in enumerate(parts):
            if i == 0 and part:  # Assistant's message
                assistant_msg = part.split("</s>")[0].strip()
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            elif part:  # User's message
                user_msg_parts = part.split("[/INST]")
                user_msg = user_msg_parts[0].strip()
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                
                if len(user_msg_parts) > 1 and user_msg_parts[1].strip():
                    assistant_msg = user_msg_parts[1].split("</s>")[0].strip()
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})

    return messages

input_file_path = './train.jsonl'
output_file_path = './train_oai.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        converted_data = {"id": data["id"], "messages": []}

        conversation_messages = parse_conversation(data["text"])
        converted_data["messages"].extend(conversation_messages)
                
        json.dump(converted_data, outfile)
        outfile.write('\n')
