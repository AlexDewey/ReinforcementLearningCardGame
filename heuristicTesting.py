# if len(self.board.p2_rows[0]) > 0:
#     for column_index, attacking_card in enumerate(self.board.p2_rows[0]):
#         for card_index, card_in_hand in enumerate(self.board.p2_hand):
#             if card_in_hand > attacking_card:
#                 action_used = ["defend", card_index, 0, column_index]
#                 break
#
p2_rows = [12, 12, 12, 12, 5, 12]
p2_hands = [3, 2, 6, 0]

defence_found = False

for column_index, attacking_card in enumerate(p2_rows):
    if defence_found:
        break
    for card_index, card_in_hand in enumerate(p2_hands):
        if card_in_hand > attacking_card:
            print("defend " + str(card_index) + " 0 " + str(column_index))
            defence_found = True
            break

print("Done")
