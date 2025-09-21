from libc.stdint cimport int32_t
import cython
import numpy as np

chess_value = {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 7, 8: 6, 9: 5, 10: 4, 11: 3, 12: 2, 13: 1}

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint check_neighboring(const (int, int) src, const (int, int) dst):
    if (src[0] == dst[0] and abs(src[1] - dst[1]) == 1):
        return True
    elif (src[1] == dst[1] and abs(src[0] - dst[0]) == 1):
        return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint check_cannon_can_eat(const int [:, :] board, const (int, int) src, const (int, int) dst):
    cdef int chess_cnt = 0
    cdef int i

    # 由於前面已經判定過移動到相鄰空格，因此這裡只會是距離一格以上的空格
    if board[dst[0], dst[1]] == 14:
        return False
    # 炮/包必須隔著一顆棋吃
    if check_neighboring(src, dst):
        return False

    if src[0] == dst[0]:  # 兩顆棋在同一個 row
        if src[1] < dst[1]:
            for i in range(src[1] + 1, dst[1]):
                if board[src[0], i] != 14:
                    chess_cnt += 1
        else:
            for i in range(dst[1] + 1, src[1]):
                if board[src[0], i] != 14:
                    chess_cnt += 1
    else:  # 兩顆棋在同一個 column
        if src[0] < dst[0]:
            for i in range(src[0] + 1, dst[0]):
                if board[i, src[1]] != 14:
                    chess_cnt += 1
        else:
            for i in range(dst[0] + 1, src[0]):
                if board[i, src[1]] != 14:
                    chess_cnt += 1

    return chess_cnt == 1

@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
@cython.nogil
cdef bint is_legal_action(const ((int, int), (int, int)) action, const int [:, :] board, str player_color):
    cdef (int, int) src = action[0]
    cdef (int, int) dst = action[1]
    cdef int src_chess = board[src[0], src[1]]
    cdef int dst_chess = board[dst[0], dst[1]]

    if src != dst:  # 移動或吃子
        if player_color == 'U':  # 雙方顏色未知時只能翻棋
            return False
        elif src_chess == 15 or src_chess == 14 or dst_chess == 15:  # 起點/終點不能是暗子且起點不能是空棋
            return False
        elif player_color == 'R':
            if 0 <= src_chess <= 6 and dst_chess == 14 and check_neighboring(src, dst):
                return True  # 終點是空格可以直接移動
            elif 7 <= src_chess <= 13 or 0 <= dst_chess <= 6:
                return False
            elif src_chess == 5:  # (C)
                return check_cannon_can_eat(board, src, dst)  # 炮要特殊判定
            elif not check_neighboring(src, dst):
                return False
            elif src_chess == 0 and dst_chess == 13:  # (K) (p)
                return False  # 帥不能吃卒
            elif src_chess == 6 and dst_chess == 7:
                return True  # 兵可以吃將
            elif chess_value[src_chess] < chess_value[dst_chess]:
                return False
        elif player_color == 'B':
            if 7 <= src_chess <= 13 and dst_chess == 14 and check_neighboring(src, dst):
                return True  # 終點是空格可以直接移動
            elif 0 <= src_chess <= 6 or 7 <= dst_chess <= 13:
                return False
            elif src_chess == 12:  # (c)
                return check_cannon_can_eat(board, src, dst)  # 包要特殊判定
            elif not check_neighboring(src, dst):
                return False
            elif src_chess == 7 and dst_chess == 6:  # (k) (P)
                return False  # 將不能吃兵
            elif src_chess == 13 and dst_chess == 0:
                return True  # 卒可以吃帥
            elif chess_value[src_chess] < chess_value[dst_chess]:
                return False
    elif src_chess != 15:
        return False  # 要翻開的那格只能是暗子

    return True  # 剩餘的皆為為合法 action


@cython.boundscheck(False)
@cython.wraparound(False)
def legal_actions_cython(list all_actions, const int [:, :] board, str player_color):
    cdef list legal_actions = []
    cdef Py_ssize_t i

    for i in range(352):
        if (is_legal_action(all_actions[i], board, player_color)):
            legal_actions.append(i)
    return legal_actions
