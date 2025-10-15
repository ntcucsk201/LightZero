#ifndef LIBCHESS_H
#define LIBCHESS_H

#include "string"

static const int BOARD_SIZE = 32;
static const int ROW_COUNT = 8;
static const int COL_COUNT = 4;

static const char finEN[] = "KkGgMmRrNnCcPpX-";

/// Move turn
enum COLOR : int {
	RED,
	BLK,
    UNKNOWN,
};

/// Structure of a move
/// source square 10 ~ 6 bit | destination squrare 5 ~ 1 bit
enum MOVE : int {
	MOVE_NULL = 1024,
};

/// Piece type
enum FIN : int {
	FIN_K = 0,
	FIN_k = 1,
	FIN_G = 2,
	FIN_g = 3,
	FIN_M = 4,
	FIN_m = 5,
	FIN_R = 6,
	FIN_r = 7,
	FIN_N = 8,
	FIN_n = 9,
	FIN_C = 10,
	FIN_c = 11,
	FIN_P = 12,
	FIN_p = 13,
    FIN_COVER = 14,
    FIN_EMPTY = 15,

	FIN_COUNT = 16,
};

inline COLOR color_of(FIN f) {
	if (f == FIN_COVER || f == FIN_EMPTY) {
		return UNKNOWN;
	}
    return COLOR(f % 2);
}

inline int from_square(MOVE m) {
    return m >> 5;
}

inline int to_square(MOVE m) {
    return m & 0x1F;
}

inline MOVE make_move(int from, int to) {
	return MOVE((from << 5) | to) ;
}

inline std::string to_string(MOVE m) {
	if (m == MOVE_NULL) {
		return "a0 a0"; /// Resign move
	}
    int from = from_square(m), to = to_square(m);
    return std::string()
         + char('a' + from / ROW_COUNT)
         + char('1' + from % ROW_COUNT)
         + " "
         + char('a' + to / ROW_COUNT)
         + char('1' + to % ROW_COUNT);
}

inline int string2square(const char *str) {
	return (str[0] - 'a') * ROW_COUNT + str[1] - '1';
}

inline FIN char2fin(char c) {
	for (int i = 0; i < FIN_COUNT; i++) {
		if (c == finEN[i]) {
			return FIN(i);
		}
	}
	return FIN_COUNT;
}

inline FIN type_of(FIN f) {
	return FIN(f & 0xE);
}

inline bool can_capture(FIN attacker, FIN victim) {
	if (attacker == FIN_COVER || attacker == FIN_EMPTY || victim == FIN_COVER) {
		return false;
	}

	if (victim == FIN_EMPTY) {
		return true;
	}

	if (color_of(attacker) == color_of(victim)) {
		return false;
	}

	attacker = type_of(attacker);
	victim = type_of(victim);

	switch (attacker) {
	case FIN_K:
		return victim != FIN_P;
	case FIN_G:
		return victim != FIN_K;
	case FIN_M:
		return victim != FIN_K && victim != FIN_G;
	case FIN_R:
		return victim != FIN_K && victim != FIN_G && victim != FIN_M;
	case FIN_N:
		return victim == FIN_N || victim == FIN_C || victim == FIN_P;
	case FIN_C:
		return false;
	case FIN_P:
		return victim == FIN_K || victim == FIN_P;
	default:
		return false;
	}
}

#endif