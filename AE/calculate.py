from tqdm import tqdm
from utils.data_utils import read_tsv, data_split

calculators = ['plus', 'minus', 'times', 'divided by']
funcs = {'plus': lambda x, y: x + y, 'minus': lambda x, y: x - y, 'times': lambda x, y: x * y, 'divided by': lambda x, y: x / y}
tokens = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", ]
word2num = {v: i for i, v in enumerate(tokens)}
tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety"]
for i, v in enumerate(tens):
    word2num[v] = (i + 2) * 10


def parse(word):
    i = 0
    word = word.lower().split()
    if 'thousand' in word:
        thousand_index = word.index('thousand')
        thousand_desc = word[:thousand_index]
        i += parse(' '.join(thousand_desc)) * 1000
        word = word[thousand_index + 1:]
    if 'hundred' in word:
        hundred_index = word.index('hundred')
        hundred_desc = word[:hundred_index]
        i += parse(' '.join(hundred_desc)) * 100
        word = word[hundred_index + 1:]
    if len(word) == 0:
        return i
    if len(word) == 1:
        if word[0] in word2num:
            i += word2num[word[0]]
        else:
            if '-' not in word[0]:
                raise ValueError('Invalid word', word[0])
            else:
                tens, ones = word[0].split('-')
                i += word2num[tens] + word2num[ones]
    else:
        raise ValueError('Invalid word redundant', word)
    return i

def calculate(sentence):
    for c in calculators:
        if c in sentence:
            calculator_use = c
    number_one, number_two = sentence.split(calculator_use)
    number_one, number_two = number_one.strip(), number_two.strip()
    i, j = parse(number_one), parse(number_two)
    return funcs[calculator_use](i, j)



if __name__ == '__main__':
    df = read_tsv('data/Train2.tsv')
    train_df, eval_df = data_split(df, 0.8)
    sentences = train_df[0].tolist()
    labels = train_df[1].tolist()
    predictions = [calculate(v) for v in tqdm(sentences)]
    print(len([v for v in zip(predictions, labels) if round(v[0], 10) == round(v[1], 10)]) / len(predictions))
    errors = []
    for i, v in enumerate(zip(sentences, predictions, labels)):
        if round(v[1], 10) != round(v[2], 10):
            errors.append(v)