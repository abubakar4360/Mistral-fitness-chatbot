import pandas as pd
import re
import emoji
from datasets import Dataset, DatasetDict

#applying data cleaning techniques
def clean_data(text):
    # replacing multiple (!, ?) ino single and (|) into (,)
    text = re.sub(r'\?+', '?', text)
    # text = re.sub('\|', ',', text)
    text = re.sub(r'!+', '!', text)

    # Replacing <3 with emoji
    text = re.sub('<3', emoji.emojize(':red_heart:', variant="emoji_type"), text)

    # Replacing :) or ;) with emojis
    text = re.sub(':\)|:-\)', emoji.emojize(':smiling_face_with_smiling_eyes:', language='alias'), text)

    # Replacing ;) or ;-) with emoji
    text = re.sub(';\)|;-\)', emoji.emojize(':winking_face:', language='alias'), text)

    # Add a period at the end of parentheses if there isn't one
    text = re.sub(r'\(\s*([^)]+)\s*([^.]?)\s*\)', r'(\1\2.)', text)

    # Remove periods and spaces before closing parenthesis
    text = re.sub('\.\)', ').', text)
    text = re.sub('s*([()])\s*', r'\1', text)

    # Remove any space between the ending sentence and punctuation/stopper
    text = re.sub(r' ([!?:;.])(?=[\s\n]|$)', r'\1', text)

    # Replace multiple dots (...) with a single dot (.)
    text = re.sub(r'\.{2,}', '.', text)

    # Replace (?.) or (!.) with (?) or (!) respectively
    text = re.sub(r'([!?])\.(?=\s|$)', r'\1 ', text)

    # remove extra spaces
    text = re.sub(r' +', ' ', text)

    # removing extra sentences
    text = re.sub(r'Below is a profile of a client using this fitness app.', '', text)
    text = re.sub(r'Following is the data for a weekly check-in the same client submitted through the app to the coach. Analysing the profile and the check-in data, provide a feedback to send over chat.','Check-in Data\n', text)

    # remove only first blank line if it exists
    text = re.sub('^(\n+)', '', text)

    # Capitalize first word of each sentence, new line and bullet points
    text = re.sub(r'(^|[.!:?]\s+)(\w)', lambda match: match.group(1) + match.group(2).capitalize(), text)  # sentence
    text = re.sub(r'(\n)(\w)', lambda match: match.group(1) + match.group(2).capitalize(), text)  # new line
    text = re.sub(r'(-\s)(\w)', lambda match: match.group(1) + match.group(2).capitalize(), text)  # bullet points

    # removing 2 blank lines or more to 1 blank line (to ensure clean structured data)
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)

    # add stopper (.) at the end of line if not present
    text = '\n'.join(
        [line.strip() + '.' if line.strip() and not line.strip().endswith(('!', ':', '?', ',', '.', '-')) else line for
         line in text.split('\n')])

    # Remove duplicate
    text = re.sub(r'(\b\w+\b)(\s+\1)+', r'\1', text)

    return text

def structure_data(organized_data):
    # Merging 2 columns for Input (Col B and COl C)
    organized_data['Input'] = organized_data['Post Payment Form Data'] + "\n" + organized_data['Check-in Data']

    # Selecting 2 cols for Input and Responses
    organized_data = organized_data[['Input', 'Final Feedback']]

    system = "As a fitness and nutritional coach, your role is to guide and support client's in achieving their fitness and nutritional goals. You will analyze their progress, provide feedback, and motivate them to make positive changes. You will use a professional and encouraging tone, avoiding technical language and unethical language. \n\nYou will begin by acknowledging their efforts and discussing their achievements, such as completing assigned workouts or making improvements to their nutrition. If client have enabled Chronometer syncing, then you would provide suggestions for further enhancing your nutrition. However, if Chronometer syncing is off, you will focus on other areas of improvement. You will address areas for improvement in a positive and motivational way, encouraging them to challenge themselves. \n\nGoal for the week: \n- Set specific goals for the upcoming week (at least 4). \n- These goals should be achievable and aligned with their overall fitness and nutritional objectives. \n- The goals should be covering both fitness-related and nutrition-related targets. \n- Be short and precise in your points. \n\nEncourage clients to send two form check videos for personal guidance and support, motivating them towards their goals. Celebrate their progress and offer tailored advise."
    text = []

    for _,row in organized_data.iterrows():
        # data = "### System: \n" + system + "\n\n### Input:\n" + row['Input'] + "\n\n### Assistant:\n" + row['Final Feedback']
        data = f"[INST] <<SYS>> {system} <</SYS>> {row['Input']} [/INST] \n{row['Final Feedback']}"
        text.append(data)

    new_data = pd.DataFrame(text, columns=['text'])
    # new_data.to_csv('train.csv')

    # Create a dataset
    dataset = Dataset.from_pandas(new_data)

    validation_ratio = 0.1
    num_validation_samples = int(len(dataset) * validation_ratio)

    train_dataset = dataset.select([i for i in range(num_validation_samples, len(dataset))])
    validation_dataset = dataset.select([i for i in range(num_validation_samples)])

    # Create DatasetDict containing train and validation sets
    data_dict = DatasetDict({"Train": train_dataset, "Validation": validation_dataset})

    return data_dict
    # data_dict.save_to_disk("Dataset")         # Save the DatasetDict to disk