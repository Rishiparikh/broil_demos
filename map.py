import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from PIL import Image
from PIL import Image, ImageDraw
import numpy as np
from matplotlib.pyplot import figure
import pickle

def draw_circle(draw, center, radius, color):
    lower_bound = tuple(np.subtract(center, radius))
    upper_bound = tuple(np.add(center, radius))
    draw.ellipse([lower_bound, upper_bound], fill=color)

fig = plt.figure(figsize=(15, 15), dpi=80)
x = []
y = []
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global x, y
    x.append(ix)
    y.append(iy)
    return x, y
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# ax = fig.add_subplot()
im = Image.open("map.png")
draw = ImageDraw.Draw(im)
draw_circle(draw, (320, 1260), 50, (255,0,0))

plt.imshow(np.array(im))
plt.show()

fig = plt.figure(figsize=(15, 15), dpi=80)
plt.imshow(np.array(im))
plt.plot(x, y, linewidth=10)
plt.show()

p = open("line7.pkl", "wb")
pickle.dump([x,y], p)
p.close()
