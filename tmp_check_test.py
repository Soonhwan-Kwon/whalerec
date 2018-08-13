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


def got_correct(matches, index, answer):
        if index >= len(matches):
            return False
        match = matches[index]
        kaggle_id = match['name']
        score = match['score']
        if score < 0.92 and answer.kaggle not in trained:
            return True
        if kaggle_id == answer.kaggle:
            return True
        return False


correct = 0
wrong = 0
for item in data:
    # print(item)
    image = os.path.basename(item['image'])
    matches = item['matches']
    answer = answers[image]
    tmp_correct = False
    for ii in range(0, 4):
        tmp_correct = got_correct(matches, ii, answer)
        if tmp_correct:
            break

    if tmp_correct:
        correct += 1
    else:
        wrong += 1

print(f"Correct:  {correct}")
print(f"Wrong:  {wrong}")
print("")

print(f"Score:  {correct / (correct + wrong)}")
