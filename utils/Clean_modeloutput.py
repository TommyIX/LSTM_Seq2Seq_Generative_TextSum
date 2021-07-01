f = open('model.txt', 'r')
f_new = open('new_model.txt', 'w')
lines = f.readlines()

for line in lines:
    words_delete_index = []   # 存放要删除的重复字符出现的位置
    words = line.split()   # 按空字符分割每一行数据
    for pos1 in range(len(words)):
        pos1_number = words[pos1+1:].count(words[pos1])   # 当前字符在每一行中出现的次数
        if pos1_number == 0:   # 如果当前字符在这一行中没有重复，则跳过
            continue
        else:
            pos2 = pos1
            for pos1_repeated_times in range(pos1_number):   # 对比每个重复出现的位置的下一个字符是否也是重复的
                pos2 = words.index(words[pos1], pos2+1, len(words))   # 找到当前字符下一次出现的位置
                if pos2 >= len(words)-1:   # 如果已经查询到这一行数据的最后一个字符，则跳过
                    continue
                else:
                    if words[pos1+1] == words[pos2+1]:   # 判断当前字符的下一个字符是否与重复出现的字符的下一个字符相等
                        words_delete_index.append(pos2)
                        words_delete_index.append(pos2+1)
                    else:
                        continue

    words_delete_index = list(set(words_delete_index))   # 去掉需要重复删除的索引
    words_delete_index.sort(reverse=True)   # 对要删除的位置索引”从大到小“排列，方便后续删除操作
    for delete_index in words_delete_index:
        del words[delete_index]

    f_new.write(' '.join(words))
    f_new.write('\n')


