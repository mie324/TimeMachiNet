import os
import matplotlib.pyplot as plt
import numpy as np

dirList = os.listdir("./data/UTKFace/unlabeled")
num_males = 0
num_females = 0
ages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
races = [0, 0, 0, 0, 0]

for file in dirList:
    file_name = file.split("_")
    file_name = file_name[0:3]
    gender = int(file_name[1])

    if gender < 0 or gender > 1:
        continue

    if gender == 0:
        num_males = num_males + 1
    else:
        num_females = num_females + 1

    try:
        race = int(file_name[2])
    except:
        continue

    if race > 4 or race < 0:
        continue
    races[race] = races[race] + 1

    try:
        age = int(file_name[0])
    except:
        continue

    if age < 0 or age > 119:
        continue
    if age <= 9:
        category = 0
    elif age <= 19:
        category = 1
    elif age <= 29:
        category = 2
    elif age <= 39:
        category = 3
    elif (age <= 49):
        category = 4
    elif (age <= 59):
        category = 5
    elif (age <= 69):
        category = 6
    elif (age <= 79):
        category = 7
    elif (age <= 89):
        category = 8
    elif (age <= 99):
        category = 9
    elif (age <= 109):
        category = 10
    elif (age <= 119):
        category = 11
    ages[category] = ages[category] + 1
print("Number of males: {}, females: {}".format(num_males, num_females))
print("Races {}".format(races))
print("Ages {}".format(ages))

x = np.arange(12)
width = 1.0

ax = plt.axes()
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100-109', '110-119']
ax.set_xticks(x)
ax.set_xticklabels(labels)

plt.bar(x, ages)
plt.title("Age distribution of UTKFace dataset")
plt.xlabel("Age group")
plt.ylabel("Frequency")
plt.show()

race_label = ["White", "Black", "Asian", "Indian", "Other"]

plt.pie(races, labels=race_label, autopct='%.1f%%', startangle=90)
plt.title("Race distribution of UTKFace dataset")
plt.show()

plt.pie([num_males, num_females], labels=["Male", "Female"], autopct='%.1f%%', startangle=90)
plt.title("Gender distribution of UTKFace dataset")
plt.show()
