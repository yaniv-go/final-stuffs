import numpy as np
import re

card_dict = {}
count = 0
colors = ('g', 'y', 'r', 'b')
for color in colors:
    for j in range(0, 9):
        card_dict[color + str(j)] = count
        count += 1
    card_dict[color + "t"] = count
    count += 1

    card_dict[color + "c"] = count
    count += 1

    card_dict[color + "s"] = count
    count += 1

    card_dict[color + "p"] = count
    count += 1

    card_dict[color + 'p2'] = count
    count += 1

card_dict['s9'] = count
count += 1

card_dict['c9'] = count

    

first = re.compile(r'((?:[1-9a-zA-Z]{2,3},?)+)-([1-9a-zA-Z]{2,3})((?:-[1-9a-zA-Z]{2,3},[+-]?\d+\.\d+)+)')
second = re.compile(r'-([1-9a-zA-Z]{2,3}),([+-]?\d+\.\d+)')

with open('C:\\Users\\yaniv\\Documents\\log.txt', 'r') as f:
    data = f.readlines()   

    input_data = np.zeros((1, 78), dtype='int8')
    output_data = np.zeros((1, 58))
    c = 0
    for d in data: 
        x = re.match(first, d)
        if x:
            cards, top_card, scores = x.groups()
            scores = re.findall(second, scores)

            floats = [float(y) for x, y in scores]
            max_float = max(floats) ; min_float = min(floats)
            if min_float != max_float: floats = [(x - min_float)/(max_float - min_float) for x in floats]
            floats = [x if x > 0.3 else 0.3 for x in floats]

            cards = cards.split(',')
            for card in cards: 
                input_data[c][card_dict[card]] = 1

            base = 58
            if top_card[0] in colors:
                input_data[c][base + colors.index(top_card[0])] = 1
                base += 4
                if (x:= top_card[1]) in [str(i) for i in range(9)]:
                    x = int(x)
                    input_data[c][base + x] = 1
                elif top_card[1:] == 't': input_data[c][base + 9] = 1 
                elif top_card[1:] == 'c': input_data[c][base + 10] = 1
                elif top_card[1:] == 's': input_data[c][base + 11] = 1
                elif top_card[1:] == 'p': input_data[c][base + 12] = 1
                elif top_card[1:] == '2p': input_data[c][base + 13] = 1
            else:
                base += 18
                if top_card[0] == 'c': input_data[c][base] = 1
                elif top_card[0] == 's': input_data[c][base + 1] = 1

            values = [card_dict['c' + x[1]] if x[0] in colors and x[1] == '9' else card_dict[x] for x, y in scores]
            for i, v in zip(values, floats):
                output_data[c][i] = v

            output_data = np.append(output_data, np.zeros((1, 58)), axis=0)
            input_data = np.append(input_data, np.zeros((1, 78)), axis=0)
            c += 1

print (output_data.shape)
print (input_data.shape)

print (input_data[0:10])
print (output_data[0:10])
