import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

losses = "./losses.csv"

loss = pd.read_csv(losses)
plt.figure()
plt.title("Losses over training epochs")
columns = ["EG_L1_loss", "G_tv_loss", "G_img_loss", "Ez_loss", "D_loss", "Dz_loss"]
for j in columns :
    for i in range(len(loss[j])):
        temp = loss[j][i][:-18]
        temp2 = float(temp[7:])
        loss[j][i] = temp2

plt.plot(loss["epoch"], loss['EG_L1_loss'], label="EG_L1_loss")
plt.plot(loss["epoch"], loss['G_tv_loss'], label="G_tv_loss")
plt.plot(loss["epoch"], loss['G_img_loss'], label="G_img_loss")
plt.plot(loss["epoch"], loss['Ez_loss'], label="Ez_loss")
plt.plot(loss["epoch"], loss['D_loss'], label="D_loss")
plt.plot(loss["epoch"], loss['Dz_loss'], label="Dz_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.savefig("losses_plot.png")
