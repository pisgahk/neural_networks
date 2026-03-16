words = open("makemore/names.txt", "r").read().splitlines()
words[:10]

print(len(words))  # How many names exist in this dataset?
print(min(len(w) for w in words))  # What is the shortest name.
print(max(len(w) for w in words))  # What is the longest name.


# ---[Creating a Bigram.]--------------------------------------------------------------------------------------------------------------

b = {}

for w in words:
    # Creating the Bigram of the beginning and last letter.
    ch = ["<S>"] + list(w) + ["<E>"]

    # For every word, I get the first character and its next.
    for ch1, ch2 in zip(ch, ch[1:]):

        # Checking the number of times that a Bigram appears.
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

        # print(ch1, ch2)

# print(b)

# ---[Sorting the Bigrams]-------------------------------------------------------------------------------------------------------------

print(sorted(b.items(), key=lambda kv: -kv[1]))

# ---[Instead of keeping this information in a Dictionary, why not store it in a 2D array?]--------------------------------------------

import torch

N = torch.zeros((28,28), dtype=torch.int32)
N

# N[1, 3] = 34


# Getting all the characters in the dataset i.e. words

# Creating a lookup table to convert the characters into integers.
chars = sorted(list(set(''.join(words))))
stoi =  {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27
stoi


for w in words:
    ch = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(ch, ch[1:]):
        # bigram = (ch1, ch2)
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2 ] += 1

N # But this looks ugly, let us visualise this more clearly.

# ---[Visualising with matplotlib]----------------------------------------------------------------------------------------------
#
# itos = {i:s for s,i in stoi.items()}
# itos
#
# import matplotlib.pyplot as plt
# %matplotlib inline
#
# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(28):
#     for j in range(28):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
# plt.axis('off');
# plt.show()
#

itos = {i:s for s,i in stoi.items()}

import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(16,16))  # ← changed
ax.imshow(N, cmap='Blues')               # ← changed

for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        ax.text(j, i, chstr, ha="center", va="bottom", color='gray')   # ← ax.text
        ax.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')  # ← ax.text

ax.axis('off')  # ← changed
plt.show()
