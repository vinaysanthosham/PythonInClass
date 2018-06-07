data= input("Enter all elements seperated by space")
words = data.split()
tuple(words)
word_len = [len(x) for x in data.split()]
tuple(word_len)
word_len_pair = list(zip(word_len,words))
sorted_list = sorted(word_len_pair)
print(sorted_list)
print(sorted_list[-1])
