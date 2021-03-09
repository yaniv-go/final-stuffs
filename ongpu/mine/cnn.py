
"""
dataset_path = "C:\\Users\\yaniv\\Documents\\datasets\\dog-breed\\"
x, y = get_dogs(dataset_path)
xb = x.reshape((-1, 64, 3, 224, 224))
yb = y.reshape((-1, 64, 120))

n = int(xb.shape[0] * 0.7)
(tx, ty), (vx, vy) = (xb[:n], yb[:n]), (xb[n:], yb[n:])

with open('model-12-01.pickle', 'rb') as f:
        c = pickle.load(f)

cProfile.run('j, jv = c.adam_momentum(7, tx[::], ty[::], vx[::], vy[::], e=1e-3, wd=0, k=32)')

y = cp.load(dataset_path + 'all-labels-shuffled.npy')

print('training test: ')
c.test(x[:2048].reshape((-1 , 32, 3, 224, 224)), y[:2048].reshape((-1, 32)))

print('validation test: ')
c.test(x[-2048:].reshape((-1 , 32, 3, 224, 224)), y[-2048:].reshape((-1, 32)))

fig, axs = plt.subplots(2)
axs[0].plot(range(len(j)), j)
axs[1].plot(range(len(jv)), jv)

plt.show()

a = input('would you like to savce data? ')
if a == 'y':
    with open('model-12-01.pickle', 'wb') as f:
        pickle.dump(c, f)
"""
"""
dataset_path = "C:\\Users\\yaniv\\Documents\\datasets\\dog-breed\\"

with open(dataset_path + 'breed-dict.pickle', 'rb') as f:
    breeds = pickle.load(f)
    breeds = {value : key for key, value in breeds.items()}

with open('model-12-01.pickle', 'rb') as f:
        c = pickle.load(f)

with open('table.txt', 'w') as f:
    pic = PIL.Image.open('table.jfif').convert('RGB')
    pic = cp.array(pic.resize((224, 224)), dtype='float32')
    pic = pic.transpose(2, 0, 1).reshape((1, 3, 224, 224))
    mean = pic.mean(axis=(2, 3)).reshape((1, 3, 1, 1))
    pic -= mean
    o = c.forward(pic)
    o = o.ravel()

    f.write('table:\n\n')

    for i in range(o.shape[0]):
        print(str(o[i]))
        f.write('%s: %s\n' % (breeds[i], str(o[i])))
"""
