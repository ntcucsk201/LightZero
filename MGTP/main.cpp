#include <stdio.h>
#include <string.h>

#include "libchess.h"
#include "MyAI.h"

#define COMMAND_NUM 19
const char* commands_name[COMMAND_NUM] = {
    "protocol_version",
    "name",
    "version",
    "known_command",
    "list_commands",
    "quit",
    "boardsize",
    "reset_board",
    "num_repetition",
    "num_moves_to_draw",
    "move",
    "flip",
    "genmove",
    "game_over",
    "ready",
    "time_settings",
    "time_left",
    "showboard",
    "init_board"
};

int main() {
    std::string write;
	char read[1024], *token;
    const char *data[100];
    int id, i;
    MyAI myai;

    // Game Loop
    do {
        write.clear();
        // read command
        fgets(read, 1024, stdin);

        printf("get= %s\n", read);
        // remove newline(\n)
        read[strlen(read) - 1] = '\0';
        // get command id
        token = strtok(read, " ");
        sscanf(token, "%d", &id);
        // get command name
        token = strtok(NULL, " ");
        // get command data
        i = 0;
        while ((token = strtok(NULL, " ")) != NULL) {
            data[i++] = token;
        }

        switch (id) {
        case 0: // protocol_version
            write = myai.GetProtocolVersion();
            break;
        case 1: // name
            write = myai.GetAIName();
            break;
        case 2: // version
            write = myai.GetAIVersion();
            break;
        case 3: // known_command
            for (i = 0; i < COMMAND_NUM; i++) {
                if (strcmp(data[0], commands_name[i]) == 0) {
                    break;
                }
            }
            write = i == COMMAND_NUM ? "false" : "true";
            break;
        case 4: // list_commands
            for (int i = 0; i < COMMAND_NUM; i++) {
                write += commands_name[i];
                write += "\n";
            }
            break;
        case 5: // quit
            break;
        case 6: // boardsize
            break;
        case 7: // reset_board
            myai.InitBoard();
            // myai.Print();
            break;
        case 8: // num_repetition
            break;
        case 9: // num_moves_to_draw
            break;
        case 10: // move
            myai.Move(string2square(data[0]), string2square(data[1]));
            // myai.Print();
            break;
        case 11: // flip
            myai.Flip(string2square(data[0]), char2fin(data[1][0]));
            // myai.Print();
            break;
        case 12: // genmove
            if (strcmp(data[0], "red") == 0) {
                myai.SetColor(RED);
            } else if (strcmp(data[0], "black") == 0) {
                myai.SetColor(BLK);
            } else {
                myai.SetColor(UNKNOWN);
            }
            write = to_string(myai.GenerateMove());
            break;
        case 13: // game_over
            printf("game_over %s\n", data[0]);
            break;
        case 14: // ready
            break;
        case 15: // time_settings
            break;
        case 16: // time_left
        {
            COLOR color = strcmp(data[0], "red") == 1 ? RED : BLK;
            int time;
            sscanf(data[1], "%d", &time);
            myai.SetTime(color, time);
            break;
        }
        case 17: // showboard
            myai.Print();
            break;
        case 18: // init_board
            break; 
        }

        /// Send result to MGTP server
        printf("=%d %s\n", id, write.c_str());
        
        fflush(stdout);
        fflush(stderr);

    } while (id != 5); // Quit if receive a quit command

    return 0;
}
