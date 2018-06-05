import random

number1 = random.randrange(0,9,1)



while True:
 number2 = input("Enter guess number in the range 0-9 : ")

 num2 = int(number2)

 if (number1 > num2):
    print('Your answer is high than required')

 elif (number1 < num2):
    print('Your answer is low than required')

 else:
    print('Your answer is PERFECT!! Congratulations!!')
    break