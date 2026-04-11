#!/usr/bin/env python3
"""
Scramble Up - A Word Scramble Game

Unscramble the jumbled letters to guess the correct word!
Features:
  - Multiple difficulty levels (Easy, Medium, Hard)
  - Hint system (limited hints per round)
  - Scoring with combo bonuses for consecutive correct answers
  - Timer-based bonus points
  - High score tracking saved to file
"""

import random
import time
import json
import os
import string

# ── Word Bank ────────────────────────────────────────────────────────────────

WORDS = {
    "easy": [
        ("cat", "A small furry pet"),
        ("dog", "Man's best friend"),
        ("sun", "The star at the center of our solar system"),
        ("hat", "You wear it on your head"),
        ("cup", "You drink from it"),
        ("bed", "You sleep on it"),
        ("fish", "It swims in water"),
        ("tree", "It has leaves and branches"),
        ("book", "You read it"),
        ("cake", "A sweet birthday treat"),
        ("rain", "Water falling from the sky"),
        ("star", "It twinkles in the night sky"),
        ("frog", "A green amphibian"),
        ("milk", "A white drink from cows"),
        ("ball", "A round toy you throw"),
        ("door", "You open it to enter a room"),
        ("lamp", "It gives you light"),
        ("bird", "It has wings and can fly"),
        ("shoe", "You wear it on your foot"),
        ("ring", "Circular jewelry for your finger"),
    ],
    "medium": [
        ("planet", "Earth is one of these"),
        ("bridge", "A structure over water"),
        ("garden", "Where flowers grow"),
        ("castle", "A king lives here"),
        ("forest", "Full of trees"),
        ("rocket", "It flies to space"),
        ("puzzle", "A game of fitting pieces together"),
        ("dragon", "A mythical fire-breathing creature"),
        ("silver", "A shiny precious metal"),
        ("frozen", "Turned to ice"),
        ("island", "Land surrounded by water"),
        ("pirate", "A sea robber"),
        ("candle", "It gives light with a flame"),
        ("monkey", "A playful primate"),
        ("tunnel", "An underground passage"),
        ("guitar", "A stringed musical instrument"),
        ("basket", "You carry things in it"),
        ("trophy", "A prize for winning"),
        ("cactus", "A spiny desert plant"),
        ("parrot", "A colorful talking bird"),
    ],
    "hard": [
        ("elephant", "The largest land animal"),
        ("treasure", "Hidden riches"),
        ("sandwich", "Bread with filling"),
        ("platform", "A raised flat surface"),
        ("absolute", "Complete and total"),
        ("backpack", "You carry it on your shoulders"),
        ("calendar", "Shows days and months"),
        ("dialogue", "A conversation between people"),
        ("envelope", "You put a letter inside it"),
        ("firework", "Explodes in colorful lights"),
        ("geometry", "Math dealing with shapes"),
        ("hydrogen", "The lightest chemical element"),
        ("kilowatt", "A unit of electrical power"),
        ("language", "A system of communication"),
        ("memories", "Things you remember"),
        ("notebook", "You write notes in it"),
        ("obstacle", "Something in your way"),
        ("passport", "Travel identification document"),
        ("Question", "An inquiry seeking an answer"),
        ("shoulder", "Body part between neck and arm"),
    ],
}

# ── High Score File ──────────────────────────────────────────────────────────

HIGH_SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scramble_high_scores.json")


def load_high_scores():
    """Load high scores from file."""
    if os.path.exists(HIGH_SCORE_FILE):
        with open(HIGH_SCORE_FILE, "r") as f:
            return json.load(f)
    return []


def save_high_scores(scores):
    """Save high scores to file."""
    with open(HIGH_SCORE_FILE, "w") as f:
        json.dump(scores, f, indent=2)


def add_high_score(name, score, difficulty):
    """Add a new high score entry and keep top 10."""
    scores = load_high_scores()
    scores.append({"name": name, "score": score, "difficulty": difficulty})
    scores.sort(key=lambda x: x["score"], reverse=True)
    scores = scores[:10]
    save_high_scores(scores)
    return scores


def display_high_scores():
    """Display the high score leaderboard."""
    scores = load_high_scores()
    print("\n" + "=" * 50)
    print("         HIGH SCORES LEADERBOARD")
    print("=" * 50)
    if not scores:
        print("  No high scores yet. Be the first!")
    else:
        print(f"  {'Rank':<6}{'Name':<15}{'Score':<10}{'Difficulty'}")
        print("  " + "-" * 44)
        for i, entry in enumerate(scores, 1):
            print(f"  {i:<6}{entry['name']:<15}{entry['score']:<10}{entry['difficulty']}")
    print("=" * 50)


# ── Scramble Logic ───────────────────────────────────────────────────────────

def scramble_word(word):
    """Scramble a word ensuring it's different from the original."""
    word_lower = word.lower()
    letters = list(word_lower)
    # Keep shuffling until the scrambled version differs from the original
    attempts = 0
    while attempts < 100:
        random.shuffle(letters)
        scrambled = "".join(letters)
        if scrambled != word_lower:
            return scrambled
        attempts += 1
    # Fallback: reverse the word
    return word_lower[::-1]


def reveal_hint(word, revealed):
    """Reveal one more letter in the word as a hint."""
    unrevealed = [i for i in range(len(word)) if i not in revealed]
    if not unrevealed:
        return revealed, None
    idx = random.choice(unrevealed)
    revealed.add(idx)
    return revealed, idx


def format_hint_display(word, revealed):
    """Show the word with revealed letters and underscores for hidden ones."""
    display = []
    for i, ch in enumerate(word.lower()):
        if i in revealed:
            display.append(ch)
        else:
            display.append("_")
    return " ".join(display)


# ── Game Display ─────────────────────────────────────────────────────────────

TITLE_ART = r"""
  ____                            _     _        _   _
 / ___|  ___ _ __ __ _ _ __ ___ | |__ | | ___  | | | |_ __
 \___ \ / __| '__/ _` | '_ ` _ \| '_ \| |/ _ \ | | | | '_ \
  ___) | (__| | | (_| | | | | | | |_) | |  __/ | |_| | |_) |
 |____/ \___|_|  \__,_|_| |_| |_|_.__/|_|\___|  \___/| .__/
                                                      |_|
"""


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def display_title():
    """Display the game title."""
    print(TITLE_ART)
    print("  Unscramble the letters to find the hidden word!")
    print("=" * 55)


def display_round_header(round_num, total_rounds, score, combo, difficulty):
    """Display information for the current round."""
    combo_str = f" (x{combo} combo!)" if combo > 1 else ""
    print(f"\n--- Round {round_num}/{total_rounds} | Score: {score}{combo_str} | {difficulty.upper()} ---")


def display_scrambled(scrambled, hint_display=None):
    """Display the scrambled word and optional hint."""
    print(f"\n  Scrambled:  [ {scrambled.upper()} ]")
    if hint_display:
        print(f"  Hint:       {hint_display}")


# ── Game Loop ────────────────────────────────────────────────────────────────

def play_round(word, hint_text, round_num, total_rounds, score, combo, difficulty, max_hints):
    """Play a single round of the game. Returns (points_earned, new_combo, skipped)."""
    scrambled = scramble_word(word)
    revealed = set()
    hints_used = 0
    start_time = time.time()

    while True:
        clear_screen()
        display_title()
        display_round_header(round_num, total_rounds, score, combo, difficulty)

        hint_display = format_hint_display(word, revealed) if revealed else None
        display_scrambled(scrambled, hint_display)

        print(f"\n  Hints remaining: {max_hints - hints_used}/{max_hints}")
        print("  Type your answer, 'hint' for a hint, or 'skip' to skip.")
        print()

        guess = input("  Your answer: ").strip().lower()

        if guess == "skip":
            print(f"\n  Skipped! The word was: {word.upper()}")
            input("  Press Enter to continue...")
            return 0, 0, True

        if guess == "hint":
            if hints_used >= max_hints:
                print("\n  No hints remaining for this word!")
                input("  Press Enter to continue...")
                continue
            revealed, idx = reveal_hint(word, revealed)
            if idx is not None:
                hints_used += 1
                print(f"\n  Hint revealed! Clue: {hint_text}")
            else:
                print("\n  All letters already revealed!")
            input("  Press Enter to continue...")
            continue

        if guess == word.lower():
            elapsed = time.time() - start_time
            # Base points
            base_points = {"easy": 100, "medium": 200, "hard": 300}[difficulty]
            # Time bonus (max 100 extra points if answered within 5 seconds)
            time_bonus = max(0, int(100 - elapsed * 5))
            # Hint penalty
            hint_penalty = hints_used * 25
            # Combo multiplier
            new_combo = combo + 1
            combo_multiplier = min(new_combo, 5)  # cap at x5
            # Total
            points = max(10, int((base_points + time_bonus - hint_penalty) * (combo_multiplier / 1.0)))

            clear_screen()
            display_title()
            print(f"\n  CORRECT! The word was: {word.upper()}")
            print(f"  Time: {elapsed:.1f}s | Base: {base_points} | Time bonus: +{time_bonus} | Hint penalty: -{hint_penalty}")
            if combo_multiplier > 1:
                print(f"  Combo multiplier: x{combo_multiplier}")
            print(f"  Points earned: {points}")
            input("\n  Press Enter to continue...")
            return points, new_combo, False
        else:
            print(f"\n  Wrong! Try again.")
            input("  Press Enter to continue...")


def select_difficulty():
    """Let the player choose difficulty."""
    while True:
        clear_screen()
        display_title()
        print("\n  Select Difficulty:\n")
        print("  [1] Easy   - Short words (3-4 letters), 3 hints per word")
        print("  [2] Medium - Medium words (5-6 letters), 2 hints per word")
        print("  [3] Hard   - Long words (7+ letters), 1 hint per word")
        print()
        choice = input("  Enter choice (1/2/3): ").strip()
        if choice == "1":
            return "easy", 3
        elif choice == "2":
            return "medium", 2
        elif choice == "3":
            return "hard", 1
        else:
            print("  Invalid choice. Try again.")
            input("  Press Enter...")


def select_rounds():
    """Let the player choose number of rounds."""
    while True:
        clear_screen()
        display_title()
        print("\n  How many rounds would you like to play?")
        print("  (Choose between 5 and 20)\n")
        try:
            rounds = int(input("  Number of rounds: ").strip())
            if 5 <= rounds <= 20:
                return rounds
            print("  Please enter a number between 5 and 20.")
        except ValueError:
            print("  Please enter a valid number.")
        input("  Press Enter...")


def main_menu():
    """Display the main menu and get player choice."""
    while True:
        clear_screen()
        display_title()
        print("\n  Main Menu:\n")
        print("  [1] Play Game")
        print("  [2] View High Scores")
        print("  [3] How to Play")
        print("  [4] Quit")
        print()
        choice = input("  Enter choice (1/2/3/4): ").strip()
        if choice in ("1", "2", "3", "4"):
            return choice
        print("  Invalid choice.")
        input("  Press Enter...")


def show_how_to_play():
    """Display game instructions."""
    clear_screen()
    display_title()
    print("""
  HOW TO PLAY
  -----------
  1. A scrambled word is shown on screen.
  2. Type your guess to unscramble it.
  3. Type 'hint' to reveal a letter (limited per word).
     - The first hint also shows a clue about the word.
  4. Type 'skip' to skip a word (breaks your combo).
  5. Earn points for correct answers:
     - Harder words are worth more base points.
     - Answer quickly for a time bonus.
     - Each consecutive correct answer increases your combo!
     - Using hints reduces your score.
  6. Try to beat the high scores!
    """)
    input("  Press Enter to return to menu...")


def play_game():
    """Run a full game session."""
    difficulty, max_hints = select_difficulty()
    total_rounds = select_rounds()

    word_pool = list(WORDS[difficulty])
    random.shuffle(word_pool)

    # If we need more rounds than available words, cycle
    if total_rounds > len(word_pool):
        extra = total_rounds - len(word_pool)
        word_pool += random.choices(WORDS[difficulty], k=extra)

    score = 0
    combo = 0
    correct = 0
    skipped = 0

    for round_num in range(1, total_rounds + 1):
        word, hint_text = word_pool[round_num - 1]
        points, combo, was_skipped = play_round(
            word, hint_text, round_num, total_rounds, score, combo, difficulty, max_hints
        )
        score += points
        if was_skipped:
            skipped += 1
        else:
            correct += 1

    # Game over summary
    clear_screen()
    display_title()
    print("\n" + "=" * 50)
    print("              GAME OVER!")
    print("=" * 50)
    print(f"\n  Difficulty:  {difficulty.upper()}")
    print(f"  Rounds:      {total_rounds}")
    print(f"  Correct:     {correct}")
    print(f"  Skipped:     {skipped}")
    print(f"  Final Score: {score}")
    print()

    # Check for high score
    high_scores = load_high_scores()
    is_high_score = len(high_scores) < 10 or score > high_scores[-1]["score"] if high_scores else True

    if is_high_score and score > 0:
        print("  NEW HIGH SCORE!")
        name = input("  Enter your name: ").strip()
        if not name:
            name = "Anonymous"
        name = name[:14]  # limit name length
        scores = add_high_score(name, score, difficulty)
        print()
        display_high_scores()
    else:
        print("  Keep playing to reach the high scores!")

    print()
    input("  Press Enter to return to menu...")


def main():
    """Main entry point for the game."""
    try:
        while True:
            choice = main_menu()
            if choice == "1":
                play_game()
            elif choice == "2":
                clear_screen()
                display_high_scores()
                input("\n  Press Enter to return to menu...")
            elif choice == "3":
                show_how_to_play()
            elif choice == "4":
                clear_screen()
                print(TITLE_ART)
                print("  Thanks for playing Scramble Up! Goodbye!")
                print()
                break
    except KeyboardInterrupt:
        print("\n\n  Thanks for playing! Goodbye!")


if __name__ == "__main__":
    main()
