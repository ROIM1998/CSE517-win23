
from utils.data_utils import get_raw_data
from collections import Counter

def swap(d, a, b):
    reverse_dict = {v: k for k, v in d.items()}
    d[reverse_dict[a]], d[reverse_dict[b]] = d[reverse_dict[b]], d[reverse_dict[a]]
    return d

if __name__ == '__main__':
    fn = '/data0/zbw/cse517/AA.encrypted.txt'
    s = open(fn, 'r').read()
    chars = [c for c in s if c.isalpha()]
    char_cnt = dict(Counter(chars))
    
    # Using the sentiment-analysis dataset to calculate character frequencies in English
    data = get_raw_data('data/txt_sentoken')
    alphas = [c.upper() for d in data for c in d[0] if c.isalpha()]
    alpha_cnt = dict(Counter(alphas))
    
    # Matching the characters of encrypted text to the characters of English text
    char_by_freq = sorted(char_cnt.items(), key=lambda x: x[1], reverse=True)
    alpha_by_freq = sorted(alpha_cnt.items(), key=lambda x: x[1], reverse=True)
    mapping = {c[0]: a[0] for c, a in zip(char_by_freq, alpha_by_freq)}
    decrypted_text = ''.join([mapping[c] if c in mapping else c for c in s])
    
    # Seems A and O are mispaced
    swap(mapping, 'A', 'O')
    # Guessing that "TLE REUOYNITION TLAT" should be "THE RECOGNITION THAT"
    swap(mapping, 'H', 'L')
    swap(mapping, 'C', 'U')
    swap(mapping, 'G', 'Y')
    swap(mapping, 'U', 'D')
    swap(mapping, 'M', 'D')
    swap(mapping, 'P', 'B')
    swap(mapping, 'Y', 'W')
    swap(mapping, 'F', 'W')
    # Check again those low-frequency characters
    swap(mapping, 'X', 'Z')
    with open('decrypted.txt', 'w') as f:
        f.write(decrypted_text)
    