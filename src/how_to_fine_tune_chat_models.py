import json
import openai
import os
import pandas as pd
from pprint import pprint

from dotenv import load_dotenv

load_dotenv(override=True)

# client = openai.OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     organization="evoagile",
#     project="FineTune",
# )
client = openai.OpenAI()

# Read in the dataset we'll use for this task.
# This will be the RecipesNLG dataset, which we've cleaned to only contain documents from www.cookbooks.com
recipe_df = pd.read_csv("data/cookbook_recipes_nlg_10k.csv")

recipe_df.head()

system_message = (
    "You are a helpful recipe assistant."
    "You are to extract the generic ingredients from each of the recipes provided."
)


def create_user_message(row):
    return f"Title: {row['title']}\n\nIngredients: {row['ingredients']}\n\nGeneric ingredients: "


def prepare_example_conversation(row):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": create_user_message(row)},
            {"role": "assistant", "content": row["NER"]},
        ]
    }


pprint(prepare_example_conversation(recipe_df.iloc[0]))

# use the first 100 rows of the dataset for training
training_df = recipe_df.loc[0:100]

# apply the prepare_example_conversation function to each row of the training_df
training_data = training_df.apply(prepare_example_conversation, axis=1).tolist()

for example in training_data[:5]:
    print(example)

validation_df = recipe_df.loc[101:200]
validation_data = validation_df.apply(prepare_example_conversation, axis=1).tolist()


def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)


training_file_name = "tmp_recipe_finetune_training.jsonl"
write_jsonl(training_data, training_file_name)

validation_file_name = "tmp_recipe_finetune_validation.jsonl"
write_jsonl(validation_data, validation_file_name)


def upload_file(file_name: str, purpose: str) -> str:
    with open(file_name, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)
    return response.id


training_file_id = upload_file(training_file_name, "fine-tune")
validation_file_id = upload_file(validation_file_name, "fine-tune")

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)

MODEL = "gpt-4o-mini-2024-07-18"

response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model=MODEL,
    suffix="recipe-ner",
)

job_id = response.id

print("Job ID:", response.id)
print("Status:", response.status)

response = client.fine_tuning.jobs.retrieve(job_id)

print("Job ID:", response.id)
print("Status:", response.status)
print("Trained Tokens:", response.trained_tokens)

response = client.fine_tuning.jobs.list_events(job_id)

events = response.data
events.reverse()

for event in events:
    print(event.message)

response = client.fine_tuning.jobs.retrieve(job_id)
fine_tuned_model_id = response.fine_tuned_model

if fine_tuned_model_id is None:
    raise RuntimeError(
        "Fine-tuned model ID not found. Your job has likely not been completed yet."
    )

print("Fine-tuned model ID:", fine_tuned_model_id)


test_df = recipe_df.loc[201:300]
test_row = test_df.iloc[0]
test_messages = []
test_messages.append({"role": "system", "content": system_message})
user_message = create_user_message(test_row)
test_messages.append({"role": "user", "content": user_message})

pprint(test_messages)

response = client.chat.completions.create(
    model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=500
)
print(response.choices[0].message.content)
