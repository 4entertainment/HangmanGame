import os
import random
from hangman_words import word_list
from hangman_art import stages, logo

# Function to clear the screen
def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

chosen_word = random.choice(word_list)
word_length = len(chosen_word)
lives = 6  # Remaining lives counter

print(logo)
print(f'Pssst, the solution is {chosen_word}.')

# Prediction list:
display = ["_"] * word_length
temp_display = []

while True:
    clear_screen()  # Clear the screen after each guess

    print(stages[lives])
    print(" ".join(display))
    guess = input("Guess a letter: ").lower()

    if not guess.isalpha() or len(guess) != 1:
        print("Please enter a valid single letter.")
        continue

    if guess in temp_display:
        print(f"You've already guessed {guess}, please continue. Remaining Lives: {lives}")
        continue

    temp_display.append(guess)

    # Check guessed letter:
    found = False
    for position in range(word_length):
        letter = chosen_word[position]
        if letter == guess:
            display[position] = letter
            found = True

    if not found:
        lives -= 1
        print(f"You guessed {guess}, that's not in the word. You lose a life. Remaining Lives: {lives}")

    if "_" not in display:
        print("You win!")
        break
    if lives == 0:
        print(stages[lives])
        print("You lose. The correct word was: " + chosen_word)
        break
