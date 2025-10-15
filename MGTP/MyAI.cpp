#include <string.h>

#include "libchess.h"
#include "MyAI.h"
using namespace std;

MyAI::MyAI() {
	InitBoard();
}

/*
 * Initial board
 */
void MyAI::InitBoard() {
	const int cover[14] = {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5};

	color = UNKNOWN;
	time[RED] = 0;
	time[BLK] = 0;
	memcpy(coverPieceCount, cover, sizeof(int) * 14);
	allCoverCount = BOARD_SIZE;
	
	for (int i = 0, sq = 0; i < ROW_COUNT; i++) {
		for (int j = 0; j < COL_COUNT; j++, sq++) {
			board[sq] = FIN_COVER;
		}
	}
}
/*
 * Initial board by giving position
 * 
 * @param data : a string array from MGTP server
 * data[0 : 31]  the piece type on the board, from left to right, up to down
 * data[32 : 45] the number of each piece type that is covered
 *               the order of piece type is "KkGgMmRrNnCcPp"
 */
void MyAI::InitBoard(const char* data[]) {
	color = UNKNOWN;
	time[RED] = 0;
	time[BLK] = 0;
	allCoverCount = BOARD_SIZE;
	for (int r = ROW_COUNT - 1, i = 0; r >= 0; r--) {
		for (int c = 0; c < COL_COUNT; c++, i++) {
			board[r + c * 4] = char2fin(data[i][0]);
		}
	}
	
	for (int i = 0; i < FIN_COVER; i++) {
		coverPieceCount[i] = data[i + 32][0] - '0';
		allCoverCount += coverPieceCount[i];
	}
}

/*
 * Move a piece and alternate move turn
 *
 * @param from : the source square of the move piece 
 * @param to : the destination square of the move piece
 */
void MyAI::Move(int from, int to) {
	if (color == RED) {
		color = BLK;
	} else if (color == BLK) {
		color = RED;
	}
	board[to] = board[from];
	board[from] = FIN_EMPTY;
}

/*
 * Flip a piece and alternate move turn
 *
 * @param sq : the square of the flip piece 
 * @param f : the piece type of the cover piece
 */
void MyAI::Flip(int sq, FIN f) {
	if (color == RED) {
		color = BLK;
	} else if (color == BLK) {
		color = RED;
	} else {
		color = !color_of(f);
	}
	board[sq] = f;
	coverPieceCount[f]--;
	allCoverCount--;
}

void MyAI::SetColor(COLOR c) {
	color = c;
}

void MyAI::SetTime(COLOR c, int t) {
	time[c] = t;
}

/*
 * Generate the best move of current player
 * This function will choose a random move so you may want to modify this.
 *
 * TODO: your work here
 */
MOVE MyAI::GenerateMove() const {
	MOVE moveList[100];	
	int moveCount = 0;
	int to, cnt, row, col;

	for (int i = 0; i < BOARD_SIZE; i++) {
		if (board[i] == FIN_COVER) {
			/// Flip a cover piece
			moveList[moveCount++] = make_move(i, i);
		} else if (board[i] != FIN_EMPTY && color_of(board[i]) == color) {
			row = i % ROW_COUNT;
			col = i / ROW_COUNT;
			if (board[i] == FIN_C || board[i] == FIN_c) {
				for (int delta : {-ROW_COUNT, +1, +ROW_COUNT, -1}) {
					to = i + delta;
					cnt = 0;
					while (to >= 0 && to < BOARD_SIZE && (to % ROW_COUNT == row || to / ROW_COUNT == col)) {
						cnt += (board[to] != FIN_EMPTY);
						if (cnt == 2 && color_of(board[to]) == !color) {
							/// Move a cannon to capture other piece
							moveList[moveCount++] = make_move(i, to);
							break;
						}
						to += delta;
					}
				}
			}
			for (int to : {i-ROW_COUNT, i+1, i+ROW_COUNT, i-1}) {
				if (to >= 0 && to < BOARD_SIZE && (to % ROW_COUNT == row || to / ROW_COUNT == col)
				 && can_capture(board[i], board[to])) {
					/// Move a piece to nearby square
					moveList[moveCount++] = make_move(i, to);
				}
			}
		}
	}

	printf("legal: ");
	for (int i = 0; i < moveCount; i++) {
		printf("%s, ", to_string(moveList[i]).c_str());
	}
	printf("\n");

	if (moveCount > 0) {
		return moveList[rand() % moveCount];
	}
	return MOVE_NULL;
}

string MyAI::GetProtocolVersion() const {
	return "1.1.0";
}

string MyAI::GetAIName() const {
	return "MyAI";
}

string MyAI::GetAIVersion() const {
	return "1.0.0";
}

/*
 * Print current position state
 */
void MyAI::Print() const {	
	if (color == RED) {
		printf("[RED] ");
	} else if (color == BLK) {
		printf("[BLK] ");
	} else {
		printf("[UNKNOWN] ");
	}

	for (int i = 0; i < FIN_COVER; i++) {
		printf("%d ", coverPieceCount[i]);
	}
	printf("\n");

	for (int i = ROW_COUNT - 1; i >= 0; i--) {
		printf("%d ", i+1);
		for (int j = 0; j < BOARD_SIZE; j += ROW_COUNT) {
			printf("%c ", finEN[board[i + j]]);
		}
		printf("\n");
	}
	printf("  a b c d\n");
}
