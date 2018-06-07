fo = open("foo.txt", "w")
fo.write( "Python is a great language.\nYeah its great!!");
fo = open("foo.txt", "r")

for item in fo:
    wordcount = 1
    for i in item:
     if i == " ":
        wordcount=wordcount+1
    print(item.strip(),',',wordcount)
fo.close()