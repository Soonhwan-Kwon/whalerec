import csv
import json
import os


class PlaygroundItem(object):
    def __init__(self, kaggle, hw):
        self.kaggle = kaggle
        self.hw = hw


answers = {}
trained = set()
with open("/Users/ken/dev/animalus/animalus-core/dev/kaggle/playground_image_map.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for row in reader:
        kaggle_id = row[3]
        if row[5] == "Train":
            trained.add(kaggle_id)
        else:
            answers[row[4]] = PlaygroundItem(kaggle_id, f"{row[1]}-{row[2]}")

# Example output
# [{"image": "/home/ubuntu/kaggle/test2/40bdf228.jpg", "matches": []}, {"image": "/home/ubuntu/kaggle/test2/408ecac5.jpg", "matches": [{"name": "w_f8c3a63", "score": 1.0}]}]
with open("id_test.json", 'r') as input:
    data = json.loads(input.read())

correct = 0
wrong = 0
nothing_found = 0
correct_new_whales = 0
no_answer = 0
for item in data:
    # print(item)
    image = os.path.basename(item['image'])
    matches = item['matches']
    if len(matches) == 0:
        # print(answer.kaggle, answer.kaggle in trained)
        if answer.kaggle in trained:
            nothing_found += 1
        else:
            correct_new_whales += 1
    else:
        answer = answers[image]
        if answer is None:
            no_answer += 1
        else:
            if matches[0]['name'] == answer.kaggle:
                correct += 1
            else:
                wrong += 1

print(f"No Answer:  {no_answer}")
print(f"Nothing Found:  {nothing_found}")
print(f"Correct new whales:  {correct_new_whales}")
print(f"Correct:  {correct}")
print(f"Wrong:  {wrong}")
print("")

totalCorrect = correct_new_whales + correct
totalWrong = nothing_found + wrong

print(f"Total Correct:  {totalCorrect}")
print(f"Total Wrong:  {totalWrong}")

print(f"Score:  {totalCorrect / (totalCorrect + totalWrong)}")
