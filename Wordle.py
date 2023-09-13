with open(r"C:\\Users\manas\Desktop\Manas\Coding\Projects\Wordle\sgb-words.txt", 'r') as file:
    words = file.read().split('\n')
    words = [word for word in words if word]
import random
number = random.randint(0, len(words))
start = ' '
correct_word = words[number]
count = 1

while count <= 5:
    if start == correct_word:
        print("Congrats you won!")
        break
    start = input('Enter a word (Try: ' + str(count) + '):')
    start = start.lower()
    if len(start) != 5:
        print("Please write a 5 letter word!")
        start = input('Enter a word (Try: ' + str(count) + '):')
        if start not in words:
            print(start + " is not a real word, TRY AGAIN!")
            start = input('Enter a word (Try: ' + str(count) + '):')
    count += 1
    for i in range(len(start)):
        if start[i] == correct_word[i]:
            print(start[i] + " is the correct letter, correct spot")
        elif start[i] in correct_word:
            print(start[i] + " is the correct letter, not in correct spot")
        else:
            print(start[i] + " is the wrong letter")
if count > 5:
    print("""Sorry, you could not get the correct word in 5 tries. 
The correct answer is: """ + correct_word)
