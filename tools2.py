import random
from typing import List, Tuple

import streamlit as st


def generate_card_pairs(num_pairs: int) -> List[Tuple[str, str]]:
  """
  Generates a list of card pairs with unique image or text content.
  """
  symbols = [f"symbol_{i}" for i in range(num_pairs)] * 2  # Duplicate list for pairs
  random.shuffle(symbols)
  return list(zip(symbols, symbols))


def display_card(symbol: str, is_flipped: bool) -> None:
  """
  Displays a card based on its symbol and flipped state.
  """
  if is_flipped:
    st.write(symbol)
  else:
    st.write("?")


def game_loop(num_pairs: int):
  """
  Main game loop that manages card states, player interaction, and feedback.
  """
  card_pairs = generate_card_pairs(num_pairs)
  flipped_cards = [False] * len(card_pairs)  # Track flipped card states
  first_card_index = None

  while True:
    # Display the grid of cards
    for i, (symbol, _) in enumerate(card_pairs):
      if i > 0:
        st.write(" ")  # Add spacing between cards
      display_card(symbol, flipped_cards[i])

    # Get player input (click on a card)
    clicked_index = st.button("Click a card1", key="card_button1")
    if clicked_index is not None:
      flipped_cards[clicked_index] = True

      # Check for match if two cards are flipped
      if first_card_index is not None and clicked_index != first_card_index:
        if card_pairs[first_card_index][0] == card_pairs[clicked_index][0]:
          st.success("Match!")
        else:
          st.error("No match. Try again.")
          # Briefly show both flipped cards before hiding them again
          st.write(card_pairs[first_card_index][0])
          st.write(card_pairs[clicked_index][0])
          st.write("---")  # Separator
          for i in range(len(card_pairs)):
            flipped_cards[i] = False if i == first_card_index or i == clicked_index else flipped_cards[i]
          st.write("---")  # Separator
        first_card_index = None

      else:
        first_card_index = clicked_index

    # Check if all cards have been matched
    if all(flipped_cards):
      st.success("Congratulations! You found all matches!")
      break


st.title("Memory Match Game")
num_pairs = st.slider("Number of Pairs", min_value=2, max_value=10, key="num_pairs")
st.button("Start Game", on_click=game_loop, args=(num_pairs,))
