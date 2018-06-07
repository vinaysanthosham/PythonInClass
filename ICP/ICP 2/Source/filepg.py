fo = open("foo.txt", "w")
fo.write( "Python\nis\na great\nlanguage.\nYeah\nits\ngreat!!\n");
fo = open("foo.txt", "r")
for t in fo:
    print(t.strip(),',',len(t.strip()))
fo.close()