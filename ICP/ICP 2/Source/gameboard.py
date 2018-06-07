heightinp= int(input("Enter the height of the board: "))
widthinp= int(input("Enter the width of the board: "))

def board_draw(h,w):
    for j in range(w):
     print('--- ',end="")
    print('\n',end="")
    for k in range(h+1):
     print('|  ',end="")
    print('\n',end="")



for i in range(heightinp):
    board_draw(heightinp,widthinp);


for l in range(widthinp):
        print('--- ', end="")